#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
import pandas as pd
import os
import random
from PIL import Image

his_buffer_len = 2 ** 11


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes * 4)
        self.Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes * 4, num_classes * 4)) for s in range(num_R)])

    def forward(self, x, fhis_in=None):
        fhis_in_PG = None
        if fhis_in:
            fhis_in_PG = fhis_in[0]
        out, fhis_out_PG = self.PG(x, fhis_in_PG)
        """
            if fhis_in_PG==None:
                out: 1x64x2048
                fhis_out_PG: [12*[1x64x2048]]
            else:
                out: 1x64x24
                fhis_out_PG: [12*[1x64x2048]]
        """
        fhis_out = []
        fhis_out.append(fhis_out_PG)
        outputs = out.unsqueeze(0)  # 1x1x64x2048  # out=1,64,5760, outputs=1,1,64,5760
        k = 1
        for R in self.Rs:
            if fhis_in:
                fhis_in_r = fhis_in[k]
                k = k + 1
                out, fhis_out_r = R(F.softmax(out, dim=1), fhis_in_r)
            else:
                out, fhis_out_r = R(F.softmax(out, dim=1))  # out: 1x64x2048, fhis_ouy_r: [11x[1x64x2048]]
            # fhis_out.append(out)
            fhis_out.append(fhis_out_r)

            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)  # 2x1x64x2048 ; 4x1x64x24

        return outputs, fhis_out


'''
#kernelsize=2
class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 2, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 2, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f_dia1=self.conv_dilated_1[i](f)
            f_dia1=f_dia1[:,:,:f.shape[2]]
            f_dia2=self.conv_dilated_2[i](f)
            f_dia2=f_dia2[:,:,:f.shape[2]]
            f = self.conv_fusion[i](torch.cat([f_dia1, f_dia2], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out
'''


# kernelsize=3
class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers - i), dilation=2 ** (num_layers - 1 - i))
            for i in range(num_layers)
        ))
        self.conv_dilated_online_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=0, dilation=2 ** (num_layers - 1 - i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (i + 1), dilation=2 ** i)
            for i in range(num_layers)
        ))
        self.conv_dilated_online_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=0, dilation=2 ** i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fhis_in=None):
        f = self.conv_1x1_in(x)  # x=1,2048,5760
        fhis_out = []
        if fhis_in:
            len_cur = f.shape[2]
            len_his = fhis_in[0].shape[2]
            f_his_new = torch.cat([fhis_in[0], f], 2)
            if len_cur + len_his > his_buffer_len:
                f_his_new = f_his_new[:, :, -his_buffer_len:]
            fhis_out.append(f_his_new)
        else:
            fhis_out.append(f)
        if fhis_in:
            for i in range(self.num_layers):
                f_in = f
                f_his_i = fhis_in[i]
                if i > 0:
                    # fea_dim=f_his_i.shape[1]
                    # f1=torch.cat([f_his_i[:,:int(fea_dim/2),-2**(self.num_layers-i):],f],2)
                    # f2=torch.cat([f_his_i[:,int(fea_dim/2):,-2**(i+1):],f],2)
                    f1 = torch.cat([f_his_i[:, :, -2 ** (self.num_layers - i):], f], 2)
                    f2 = torch.cat([f_his_i[:, :, -2 ** (i + 1):], f], 2)
                else:
                    f1 = torch.cat([f_his_i[:, :, -2 ** (self.num_layers - i):], f], 2)
                    f2 = torch.cat([f_his_i[:, :, -2 ** (i + 1):], f], 2)
                f_dia1 = self.conv_dilated_online_1[i](f1)
                f_dia2 = self.conv_dilated_online_2[i](f2)
                f_dia = torch.cat([f_dia1, f_dia2], 1)
                f = self.conv_fusion[i](f_dia)
                f = F.relu(f)
                f = self.dropout(f)
                f = f + f_in

                f_his_i1 = fhis_in[i + 1]
                len_cur = f.shape[2]
                len_his = f_his_i1.shape[2]
                f_his_new_i = torch.cat([f_his_i1, f], 2)
                if len_cur + len_his > his_buffer_len:
                    f_his_new_i = f_his_new_i[:, :, -his_buffer_len:]
                fhis_out.append(f_his_new_i)
        else:
            for i in range(self.num_layers):
                f_in = f
                f_dia1 = self.conv_dilated_1[i](f)  # f=1,64,5760,fdia1=1,64,7808
                # f_dia1=f_dia1[:,:,2**(self.num_layers-i):]#fdia1=1,64,7808
                f_dia1 = f_dia1[:, :, :f.shape[2]]
                f_dia2 = self.conv_dilated_2[i](f)
                # f_dia2=f_dia2[:,:,2**(i+1):]
                f_dia2 = f_dia2[:, :, :f.shape[2]]
                f_dia = torch.cat([f_dia1, f_dia2], 1)
                f = self.conv_fusion[i](f_dia)
                f = F.relu(f)
                f = self.dropout(f)
                f = f + f_in
                fhis_out.append(f)

        out = self.conv_out(f)

        return out, fhis_out


class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fhis_r=None):
        fhis_out = []
        out = self.conv_1x1(x)
        if fhis_r:
            len_cur = out.shape[2]
            len_his = fhis_r[0].shape[2]
            f_his_new = torch.cat([fhis_r[0], out], 2)
            if len_cur + len_his > his_buffer_len:
                f_his_new = f_his_new[:, :, -his_buffer_len:]
            fhis_out.append(f_his_new)
            k = 0
            for layer in self.layers:
                fhis_in_r = fhis_r[k]
                k = k + 1
                out = layer(out, [fhis_in_r])

                len_cur = out.shape[2]
                fhis_in_r1 = fhis_r[k]
                len_his = fhis_in_r1.shape[2]
                f_his_new1 = torch.cat([fhis_in_r1, out], 2)
                if len_cur + len_his > his_buffer_len:
                    f_his_new1 = f_his_new1[:, :, -his_buffer_len:]
                fhis_out.append(f_his_new1)
            out = self.conv_out(out)

        else:
            fhis_out.append(out)
            for layer in self.layers:
                out = layer(out)
                fhis_out.append(out)
            out = self.conv_out(out)

        return out, fhis_out


'''    
class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out
'''
'''
#kernel=2
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 2, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = out[:,:,:x.shape[2]]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out
'''


# kernel=3
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=2 * dilation, dilation=dilation)
        self.conv_dilated_online = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.dilation = dilation

    def forward(self, x, his_in=None):
        if his_in:
            x0 = torch.cat([his_in[0][:, :, -2 * self.dilation:], x], 2)
            dia_out = self.conv_dilated_online(x0)
            out = F.relu(dia_out)
            out = self.conv_1x1(out)
            out = self.dropout(out)
        else:
            dia_out = self.conv_dilated(x)
            out = F.relu(dia_out)
            # out = out[:,:,2*self.dilation:]
            out = out[:, :, :x.shape[2]]
            out = self.conv_1x1(out)
            out = self.dropout(out)
        return x + out


def extract_action_interval(numActions, action_res):
    action_intervals = [[] for x in range(numActions)]
    start = 0
    for i in range(len(action_res) - 1):
        if action_res[i] != action_res[i + 1]:
            end = i
            action_intervals[action_res[i]].append([start, end])
            start = i + 1
    end = len(action_res) - 1
    action_intervals[action_res[end]].append([start, end])
    return action_intervals


def acc_action_intervals(numActions, list_pred, list_gt):
    res = np.zeros([numActions, 2])
    for k in range(numActions):
        lk_pred = list_pred[k]
        lk_gt = list_gt[k]
        num_pred = len(lk_pred)
        num_correct = 0
        threshold = 0.1
        for p in lk_pred:
            for g in lk_gt:
                union = max(g[1], p[1]) - min(g[0], p[0])
                inter = max(0, min(g[1], p[1]) - max(g[0], p[0]))
                iou = inter / union
                if iou > threshold:
                    num_correct = num_correct + 1
                    break
        num_gt = len(lk_gt)
        num_recall = 0
        for p in lk_gt:
            for g in lk_pred:
                union = max(g[1], p[1]) - min(g[0], p[0])
                inter = max(0, min(g[1], p[1]) - max(g[0], p[0]))
                iou = inter / union
                if iou > threshold:
                    num_recall = num_recall + 1
                    break
        if num_gt > 0:
            recall = num_recall / num_gt
        else:
            recall = -1
        if num_pred > 0:
            precsion = num_correct / num_pred
        else:
            precsion = -1
        res[k] = [precsion, recall]
        if precsion >= 0 and precsion < 1:
            aa = 0
        if recall >= 0 and recall < 1:
            bb = 0
    return res


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split):
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
        # set the weight for particular label, follow the oder on mapping.txt

        background_weight = 0.02
        # take_put_weight = 2

        weights = [background_weight, background_weight]
        for _ in range(num_classes - 2):
            weights.append((1 - 2 * background_weight) / (num_classes - 2))
        '''
        weights = [0.0]
        for _ in range(num_classes-1):
            weights.append((1-background_weight)/(num_classes-1))
        '''
        # import pdb
        # pdb.set_trace()
        weights = torch.tensor(weights)
        # weights[10:14]=weights[10:14]*take_put_weight
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)

        '''
        weights = [0,0.04,0.066,0.071,0.061,0.07,0.066,0.07,1,0.98,0.07,0.07,0.07,0.07,0.065,0.07]
        weights = torch.tensor(weights)
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
        '''
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss(reduction='none')
        # set the weight of mseloss
        self.delta = 0.15
        self.num_classes = num_classes

        logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions, fhis = self.model(batch_input)

                predictions = predictions[:, :, :, his_buffer_len:]
                batch_target = batch_target[:, his_buffer_len:]
                mask = mask[:, :, his_buffer_len:]

                # import pdb
                # pdb.set_trace()

                loss = 0
                for pp in predictions:
                    # pp=1,16*4,5800
                    for t in range(4):
                        weight_loss = 1
                        if t > 0:
                            weight_loss = 0.8
                        p = pp[:, self.num_classes * t:self.num_classes * (t + 1), :]
                        batch_target_p = batch_target.clone()
                        batch_target_p[:, :batch_target.shape[1] - 8 * t] = batch_target[:, 8 * t:]
                        loss += weight_loss * self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target_p.view(-1))
                        loss += weight_loss * self.delta * torch.mean(
                            torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                                        max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                prediction_prob = predictions[-1]
                _, predicted = torch.max(prediction_prob[:, :self.num_classes, :].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            if (epoch + 1) % 50 == 0:
                for key in self.model.state_dict().keys():
                    if 'dilated_online' in key:
                        key_offline = key.replace('_online', '')
                        # print(key)
                        self.model.state_dict()[key].copy_(self.model.state_dict()[key_offline])

                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                                     float(correct) / total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, gt_path, epoch, actions_dict, device, sample_rate):
        i3d_frame_len = 16
        class_dict = dict((value, key) for key, value in actions_dict.items())
        self.model.eval()
        # for debug
        raw_img_path = '/disk1/dataset/ping/mythwaredata/mstcnformat/raw_data/'
        actions_dict_revert = {}
        names_action = [i for i in actions_dict.keys()]
        ids_action = [i for i in actions_dict.values()]
        for k in range(len(names_action)):
            actions_dict_revert[ids_action[k]] = names_action[k]

        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            # list_of_vids = file_ptr.read().split('\n')[:-1]
            list_of_vids = file_ptr.read().split('\n')
            if list_of_vids[-1] == '':
                list_of_vids = list_of_vids[:-1]
            file_ptr.close()
            confuse_matrx_all = np.zeros([len(class_dict), len(class_dict)])
            confuse_matrx_acc_all = np.zeros([len(class_dict), len(class_dict)])
            confuse_matrx_recall_all = np.zeros([len(class_dict), len(class_dict)])
            iou_res_all = np.zeros([len(class_dict), 2])
            iou_res_all_valid = np.zeros([len(class_dict), 2])
            for vid in list_of_vids:
                # for debug
                f_out = open(vid[:-4] + '.txt', 'w')
                video_path1 = os.path.join(raw_img_path, vid[:-4] + '_1')
                video_path2 = os.path.join(raw_img_path, vid[:-4] + '_2')

                # print vid
                confuse_matrx = np.zeros([len(class_dict), len(class_dict)])
                confuse_matrx_acc = np.zeros([len(class_dict), len(class_dict)])
                confuse_matrx_recall = np.zeros([len(class_dict), len(class_dict)])
                features = np.load(features_path + vid.split('.')[0] + '.npy')

                file_ptr_gt = open(gt_path + vid, 'r')
                content = file_ptr_gt.read().split('\n')[:-1]
                classes = np.zeros(min(np.shape(features)[1], len(content)), dtype=np.int)
                for i in range(len(classes)):
                    classes[i] = int(actions_dict[content[i + i3d_frame_len]])

                features = features[:, ::sample_rate]

                # initilization
                features_0 = np.zeros([features.shape[0], his_buffer_len])
                features = np.concatenate([features_0, features], 1)

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions, his_fea = self.model(input_x)

                predictions = predictions[:, :, :, his_buffer_len:]

                predictions = predictions[:, :, :self.num_classes, :]
                predictions_prob = F.softmax(predictions[-1], 1)
                predicted_prob, predicted = torch.max(predictions_prob.data, 1)

                predicted = predicted.squeeze()
                predicted_prob = predicted_prob.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate(
                        (recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate))

                predicted_np = predicted.cpu().numpy()
                predicted_prob_np = predicted_prob.cpu().numpy()
                predicted_intervals = extract_action_interval(len(class_dict), predicted_np)
                gt_intervals = extract_action_interval(len(class_dict), classes)
                iou_res = acc_action_intervals(len(class_dict), predicted_intervals, gt_intervals)
                iou_res_valid = iou_res >= 0
                iou_res_all = iou_res_all + iou_res * iou_res_valid
                iou_res_all_valid = iou_res_all_valid + iou_res_valid

                frame_acc = (predicted_np == classes).sum() / classes.shape[0]
                for i in range(len(class_dict)):
                    class_id = i
                    class_name = class_dict[i]
                    index_cls = (classes == class_id)
                    if index_cls.sum() > 0:
                        cls_recal = (predicted_np[index_cls] == i).sum() / index_cls.sum()
                    else:
                        cls_recal = -1
                    index_pred = (predicted_np == class_id)
                    if index_pred.sum() > 0:
                        cls_acc = (classes[index_pred] == i).sum() / index_pred.sum()
                    else:
                        cls_acc = -1
                    print('class:{},recall:{},acc:{}'.format(class_name, cls_recal, cls_acc))
                print('\n')

                for t in range(len(predicted_np)):
                    confuse_matrx[classes[t], predicted_np[t]] = confuse_matrx[classes[t], predicted_np[t]] + 1
                    if classes[t] != predicted_np[t]:
                        f_out.writelines(
                            't={},gt={},pred={}\n'.format(t, actions_dict_revert[classes[t]], actions_dict_revert[predicted_np[t]]))
                        # print('t={},gt={},pred={}'.format(t,actions_dict_revert[classes[t]],actions_dict_revert[predicted_np[t]]))

                '''
                for i in range(len(class_dict)):#gt
                    for j in range(len(class_dict)):#predict
                        gt_value=(classes==i)#gt
                        pred_value=(predicted_np==j)#predict
                        confuse_matrx[i,j]=(gt_value*pred_value).sum()
                '''
                f_out.close()
                f_name = vid.split('/')[-1].split('.')[0]
                confuse_matrx_all = confuse_matrx_all + confuse_matrx
                confuse_matrx_recall = confuse_matrx
                confuse_matrx_recall = (confuse_matrx_recall.T / confuse_matrx.sum(1)).T
                confuse_matrx_acc = confuse_matrx
                confuse_matrx_acc = confuse_matrx_acc / confuse_matrx.sum(0)
                # np.savetxt(results_dir + "/" + f_name+'_'+str(epoch)+'_acc.txt',confuse_matrx_acc,fmt='%.2f',delimiter='\t')
                # np.savetxt(results_dir + "/" + f_name+'_'+str(epoch)+'_recall.txt',confuse_matrx_recall,fmt='%.2f',delimiter='\t')

                acc_pd = pd.DataFrame(confuse_matrx_acc)
                recall_pd = pd.DataFrame(confuse_matrx_recall)
                iou_pd = pd.DataFrame(iou_res)
                confuse_pd = pd.DataFrame(confuse_matrx)
                writer = pd.ExcelWriter(results_dir + "/" + f_name + '_' + str(epoch) + '.xlsx')
                acc_pd.to_excel(writer, 'accuracy', float_format='%.2f')
                recall_pd.to_excel(writer, 'recall', float_format='%.2f')
                iou_pd.to_excel(writer, 'iou', float_format='%.2f')
                confuse_pd.to_excel(writer, 'confuse_matrix')
                writer.save()
                writer.close()

                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
                np.save(results_dir + "/" + f_name + '_prob', predicted_prob_np)

            confuse_matrx_recall_all = confuse_matrx_all
            confuse_matrx_recall_all = (confuse_matrx_recall_all.T / confuse_matrx_all.sum(1)).T
            confuse_matrx_acc_all = confuse_matrx_all
            confuse_matrx_acc_all = confuse_matrx_acc_all / confuse_matrx_all.sum(0)
            # np.savetxt(results_dir + '/avg_acc_'+str(epoch)+'.txt',confuse_matrx_acc_all,fmt='%.2f',delimiter='   ')
            # np.savetxt(results_dir + '/avg_recall_'+str(epoch)+'.txt',confuse_matrx_recall_all,fmt='%.2f',delimiter=' ')
            iou_res_all = iou_res_all / iou_res_all_valid
            acc_pd = pd.DataFrame(confuse_matrx_acc_all)
            recall_pd = pd.DataFrame(confuse_matrx_recall_all)
            iou_pd = pd.DataFrame(iou_res_all)
            confuse_pd = pd.DataFrame(confuse_matrx_all)
            writer = pd.ExcelWriter(results_dir + '/avg_res_' + str(epoch) + '.xlsx')
            acc_pd.to_excel(writer, 'accuracy', float_format='%.2f')
            recall_pd.to_excel(writer, 'recall', float_format='%.2f')
            iou_pd.to_excel(writer, 'iou', float_format='%.2f')
            confuse_pd.to_excel(writer, 'confuse_matrix')
            writer.save()
            writer.close()

    def predict_online(self, model_dir, results_dir, features_path, vid_list_file, gt_path, epoch, actions_dict, device, sample_rate):
        i3d_frame_len = 16
        class_dict = dict((value, key) for key, value in actions_dict.items())
        self.model.eval()
        # for debug
        raw_img_path = '/disk1/dataset/ping/mythwaredata/mstcnformat/raw_data/'
        actions_dict_revert = {}
        names_action = [i for i in actions_dict.keys()]
        ids_action = [i for i in actions_dict.values()]
        for k in range(len(names_action)):
            actions_dict_revert[ids_action[k]] = names_action[k]

        with torch.no_grad():
            self.model.to(device)
            # self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            # list_of_vids = file_ptr.read().split('\n')[:-1]
            list_of_vids = file_ptr.read().split('\n')
            if list_of_vids[-1] == '':
                list_of_vids = list_of_vids[:-1]
            file_ptr.close()
            confuse_matrx_all = np.zeros([len(class_dict), len(class_dict)])
            confuse_matrx_acc_all = np.zeros([len(class_dict), len(class_dict)])
            confuse_matrx_recall_all = np.zeros([len(class_dict), len(class_dict)])
            iou_res_all = np.zeros([len(class_dict), 2])
            iou_res_all_valid = np.zeros([len(class_dict), 2])
            for vid in list_of_vids:
                # for debug
                f_out = open(vid[:-4] + '.txt', 'w')
                video_path1 = os.path.join(raw_img_path, vid[:-4] + '_1')
                video_path2 = os.path.join(raw_img_path, vid[:-4] + '_2')

                # print vid
                confuse_matrx = np.zeros([len(class_dict), len(class_dict)])
                confuse_matrx_acc = np.zeros([len(class_dict), len(class_dict)])
                confuse_matrx_recall = np.zeros([len(class_dict), len(class_dict)])
                features = np.load(features_path + vid.split('.')[0] + '.npy')  # 1024*6350

                file_ptr_gt = open(gt_path + vid, 'r')
                content = file_ptr_gt.read().split('\n')[:-1]
                classes = np.zeros(min(np.shape(features)[1], len(content)), dtype=np.int)
                for i in range(len(classes)):
                    classes[i] = int(actions_dict[content[i + i3d_frame_len]])

                features = features[:, ::sample_rate]

                # initialization
                features_0 = np.zeros([features.shape[0], his_buffer_len])
                input_0 = torch.tensor(features_0, dtype=torch.float)
                input_0.unsqueeze_(0)
                input_0 = input_0.to(device)  # 1x2048x2048
                _, fea_his = self.model(input_0)

                video_len = features.shape[1]
                buffer_in_len = 24
                features_t = np.zeros([features.shape[0], buffer_in_len])  # 2048x24
                vaild_fea_len = 0
                # fea_his=None
                recognition = []
                predicted_np = np.zeros(video_len)
                predicted_prob_np = np.zeros(video_len)
                idx = 0
                import time
                start = time.time()
                for t in range(video_len):

                    if vaild_fea_len < buffer_in_len - 1:
                        features_t[:, vaild_fea_len] = features[:, t]
                        vaild_fea_len = vaild_fea_len + 1
                        if t < video_len - 1:
                            continue
                    if t < video_len - 1:
                        features_t[:, vaild_fea_len] = features[:, t]
                        vaild_fea_len = vaild_fea_len + 1

                    input_x = torch.tensor(features_t, dtype=torch.float)  # 2048x24
                    input_x.unsqueeze_(0)
                    input_x = input_x.to(device)  # 1x2048x24

                    predictions_t, fea_his = self.model(input_x, fea_his)
                    """
                        predictions_t --> 4x1x64x24
                        fea_his --> [12*[1x64x2048], 11*[1x64x2048], 11*[1x64x2048], 11*[1x64x2048]]
                    """
                    predictions_t = predictions_t[:, :, :self.num_classes, :]  # 4x1x16x24
                    predictions_t_prob = F.softmax(predictions_t[-1], 1)  # 1x16x24
                    predicted_t_prob, predicted_t = torch.max(predictions_t_prob.data, 1)
                    """
                       predicted_t_prob --> 24
                       predicted_t --> 24
                    """

                    predicted_t = predicted_t.squeeze()
                    predicted_t_prob = predicted_t_prob.squeeze()

                    predicted_t = predicted_t[:vaild_fea_len]
                    predicted_t_prob = predicted_t_prob[:vaild_fea_len]

                    predicted_np[idx:idx + vaild_fea_len] = predicted_t.cpu().numpy()
                    predicted_prob_np[idx:idx + vaild_fea_len] = predicted_t_prob.cpu().numpy()
                    idx = idx + vaild_fea_len
                    vaild_fea_len = 0

                    for i in range(vaild_fea_len):
                        recognition = np.concatenate((recognition, [
                            list(actions_dict.keys())[list(actions_dict.values()).index(predicted_t[i].item())]] * sample_rate))
                end = time.time()
                print('time={}'.format((end - start) / video_len))
                predicted_np = predicted_np.astype(np.int)
                predicted_intervals = extract_action_interval(len(class_dict), predicted_np)
                gt_intervals = extract_action_interval(len(class_dict), classes)
                iou_res = acc_action_intervals(len(class_dict), predicted_intervals, gt_intervals)
                iou_res_valid = iou_res >= 0
                iou_res_all = iou_res_all + iou_res * iou_res_valid
                iou_res_all_valid = iou_res_all_valid + iou_res_valid

                frame_acc = (predicted_np == classes).sum() / classes.shape[0]
                for i in range(len(class_dict)):
                    class_id = i
                    class_name = class_dict[i]
                    index_cls = (classes == class_id)
                    if index_cls.sum() > 0:
                        cls_recal = (predicted_np[index_cls] == i).sum() / index_cls.sum()
                    else:
                        cls_recal = -1
                    index_pred = (predicted_np == class_id)
                    if index_pred.sum() > 0:
                        cls_acc = (classes[index_pred] == i).sum() / index_pred.sum()
                    else:
                        cls_acc = -1
                    print('class:{},recall:{},acc:{}'.format(class_name, cls_recal, cls_acc))
                print('\n')

                for t in range(len(predicted_np)):
                    confuse_matrx[classes[t], predicted_np[t]] = confuse_matrx[classes[t], predicted_np[t]] + 1
                    if classes[t] != predicted_np[t]:
                        f_out.writelines(
                            't={},gt={},pred={}\n'.format(t, actions_dict_revert[classes[t]], actions_dict_revert[predicted_np[t]]))
                        # print('t={},gt={},pred={}'.format(t,actions_dict_revert[classes[t]],actions_dict_revert[predicted_np[t]]))

                '''
                for i in range(len(class_dict)):#gt
                    for j in range(len(class_dict)):#predict
                        gt_value=(classes==i)#gt
                        pred_value=(predicted_np==j)#predict
                        confuse_matrx[i,j]=(gt_value*pred_value).sum()
                '''
                f_out.close()
                f_name = vid.split('/')[-1].split('.')[0]
                confuse_matrx_all = confuse_matrx_all + confuse_matrx
                confuse_matrx_recall = confuse_matrx
                confuse_matrx_recall = (confuse_matrx_recall.T / confuse_matrx.sum(1)).T
                confuse_matrx_acc = confuse_matrx
                confuse_matrx_acc = confuse_matrx_acc / confuse_matrx.sum(0)
                # np.savetxt(results_dir + "/" + f_name+'_'+str(epoch)+'_acc.txt',confuse_matrx_acc,fmt='%.2f',delimiter='\t')
                # np.savetxt(results_dir + "/" + f_name+'_'+str(epoch)+'_recall.txt',confuse_matrx_recall,fmt='%.2f',delimiter='\t')

                acc_pd = pd.DataFrame(confuse_matrx_acc)
                recall_pd = pd.DataFrame(confuse_matrx_recall)
                iou_pd = pd.DataFrame(iou_res)
                confuse_pd = pd.DataFrame(confuse_matrx)
                writer = pd.ExcelWriter(results_dir + "/" + f_name + '_' + str(epoch) + '.xlsx')
                acc_pd.to_excel(writer, 'accuracy', float_format='%.2f')
                recall_pd.to_excel(writer, 'recall', float_format='%.2f')
                iou_pd.to_excel(writer, 'iou', float_format='%.2f')
                confuse_pd.to_excel(writer, 'confuse_matrix')
                writer.save()
                writer.close()

                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
                np.save(results_dir + "/" + f_name + '_prob', predicted_prob_np)

            confuse_matrx_recall_all = confuse_matrx_all
            confuse_matrx_recall_all = (confuse_matrx_recall_all.T / confuse_matrx_all.sum(1)).T
            confuse_matrx_acc_all = confuse_matrx_all
            confuse_matrx_acc_all = confuse_matrx_acc_all / confuse_matrx_all.sum(0)
            # np.savetxt(results_dir + '/avg_acc_'+str(epoch)+'.txt',confuse_matrx_acc_all,fmt='%.2f',delimiter='   ')
            # np.savetxt(results_dir + '/avg_recall_'+str(epoch)+'.txt',confuse_matrx_recall_all,fmt='%.2f',delimiter=' ')
            iou_res_all = iou_res_all / iou_res_all_valid
            acc_pd = pd.DataFrame(confuse_matrx_acc_all)
            recall_pd = pd.DataFrame(confuse_matrx_recall_all)
            iou_pd = pd.DataFrame(iou_res_all)
            confuse_pd = pd.DataFrame(confuse_matrx_all)
            writer = pd.ExcelWriter(results_dir + '/avg_res_' + str(epoch) + '.xlsx')
            acc_pd.to_excel(writer, 'accuracy', float_format='%.2f')
            recall_pd.to_excel(writer, 'recall', float_format='%.2f')
            iou_pd.to_excel(writer, 'iou', float_format='%.2f')
            confuse_pd.to_excel(writer, 'confuse_matrix')
            writer.save()
            writer.close()


def seg_model_initialize(model_path, opt, gpu_id=1):
    """
    Initialization operations for the temporal segmentation model
    Args:
        model_path: 预训练模型路径
        gpu_id: torch的分割模型GPU-id号
        opt: 网络配置参数(easydict)

    Returns: 初始化完毕的分割模型

    """
    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(gpu_id)

    model = MS_TCN2(num_layers_PG=opt.num_layers_PG, num_layers_R=opt.num_layers_R, num_R=opt.num_R, num_f_maps=opt.num_f_maps,
                    dim=2 * opt.embed_dim, num_classes=opt.num_classes)

    model.cuda()
    model.load_state_dict(torch.load(model_path))

    return model
