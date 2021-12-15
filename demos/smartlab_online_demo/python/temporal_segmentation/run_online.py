# -*- coding: utf-8 -*-

from formatting import reformat_prediction
import os
import argparse
import cv2
import numpy as np
import torch
from collections import OrderedDict
from config_online import online_opt as opt
from feature_embedding import embed_model_initialization, embed_model_inference
from mstcn import seg_model_initialization


def reformat_pred(path_in, path_out):
    reformat_prediction(path_in, path_out)


def video_capture(capture_address, view, save_dir):
    """
    实时捕获并保存视频到指定路径
    Args:
        capture_address：监控设备id或地址
        view: FrontView or TopView
        save_dir: 图像输出路径
    """
    cap = cv2.VideoCapture(capture_address)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(filename=os.path.join(save_dir, "_".join([view, "frame", str(frame_count).zfill(4)]) + ".png"),
                        img=frame)
            frame_count += 1


def seg_decider(temporal_counts, all_embed_features):
    """
    根据特征提取进度判断是否需要执行分割
    Args:
        temporal_counts: 纪录每一帧被分类的累计次数
        all_embed_features: 特征提取进度

    Returns: list of start_idx for temporal segmentation

    """
    a = np.argwhere(temporal_counts == 0)[0].item()  # 待分割的起始点
    b = all_embed_features.shape[1]  # embedding完成了多少帧
    # print(a, b)
    seg_point_list = []
    if b > a:
        for frame_idx in range(a - opt.seg_window_step, b):
            if (frame_idx + 1) >= opt.seg_window_length:  # 到达分割最小单元
                if (frame_idx + 1 - opt.seg_window_length) % opt.seg_window_step == 0:  # 到达分割点
                    start_idx = frame_idx - opt.seg_window_length + 1
                    end_idx = start_idx + opt.seg_window_length
                    if end_idx <= b:
                        seg_point_list.append(start_idx)
                        temporal_counts[start_idx:end_idx] += 1
    return seg_point_list, temporal_counts


def main(args):
    # ======================== Embedding Model Initialization ========================
    print("Loading I3D model for feature embedding...")
    embed_batch_size = opt.embed_batch_size
    embed_window_length = opt.embed_window_length
    embed_model, embed_sess = embed_model_initialization(batch_size=embed_batch_size, embed_frame_length=embed_window_length)

    # ======================== MSTCN++ Model Initialization ========================
    print("Loading MSTCN++ model for temporal segmentation...")
    if args.front_view:
        seg_model_path = os.path.join(opt.seg_model_dir, 'split_3', "epoch-95.model")
    elif args.top_view:
        seg_model_path = os.path.join(opt.seg_model_dir, 'split_4', "epoch-95.model")
    else:
        print("Unsupported View Mode!")

    seg_model = seg_model_initialization(model_path=seg_model_path, opt=opt, gpu_id=1)
    seg_model.eval()

    # ======================== Variable Buffer ========================
    image_buffer = OrderedDict()
    all_embed_features = np.zeros((opt.embed_dim, 0))  # ndarray for saving embeddings
    max_length = 1000000
    temporal_seg_counts = np.zeros(max_length)  # 用于每帧图像的分类次数计数
    all_seg_logits = np.zeros((max_length, opt.num_classes))

    # ======================== Video Processor  ========================
    cap = cv2.VideoCapture(args.input_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()  # frame:480 x 640 x 3
        if ret:
            resize_img = cv2.resize(frame, (opt.img_size_h, opt.img_size_w))
            frame_count += 1

            image_buffer[frame_count] = resize_img

            if frame_count < embed_window_length:  # todo padding or assigning unknown label?
                print("Frame embed:", frame_count)
                padding_length = embed_window_length - frame_count
                input_data = [
                    [image_buffer[1] for i in range(padding_length)] + [image_buffer[j] for j in range(1, frame_count + 1)]
                    for k in range(embed_batch_size)]

                # ======================== Video Embedding Inference by I3D ========================
                embed_features = embed_model_inference(embed_sess=embed_sess, embed_model=embed_model,
                                                       input_data=input_data)  # ndarray: C x B

                all_embed_features = np.concatenate([all_embed_features, embed_features[:, 0:1]],
                                                    axis=1)  # ndarray: C x (embed_batch_size*num_batch)

            else:
                print("Frame embed:", frame_count)
                start_frame = all_embed_features.shape[1] + 1
                num_batch = (frame_count - start_frame + 1) // embed_batch_size

                for batch_idx in range(num_batch):
                    input_data = [
                        [image_buffer[start_frame - (embed_window_length - 1) + batch_idx * embed_batch_size + i + j] for j in
                         range(embed_window_length)] for i in range(embed_batch_size)]  # B x N x H x W x 3

                    # ======================== Video Embedding Inference by I3D ========================
                    embed_features = embed_model_inference(embed_sess=embed_sess, embed_model=embed_model,
                                                           input_data=input_data)  # ndarray: C x B

                    all_embed_features = np.concatenate([all_embed_features, embed_features],
                                                        axis=1)  # ndarray: C x (embed_batch_size*num_batch)

            # ======================== Temporal Segmentation Inference by MSTCN ========================
            seg_point_list, temporal_counts = seg_decider(temporal_seg_counts, all_embed_features)
            if len(seg_point_list) != 0:

                num_feature_clips = len(seg_point_list)
                seg_feature_list = [all_embed_features[:, idx:idx + opt.seg_window_length] for idx in seg_point_list]
                with torch.no_grad():
                    input_seg = torch.tensor(seg_feature_list, dtype=torch.float)  # B x C x N_seg 8x1024x10
                    input_seg = input_seg.cuda()
                    logits = seg_model(input_seg)  # 4 x B x K x N_seg

                    probs = torch.softmax(logits[-1].data, dim=1)  # B x K x N_seg
                    _, predictions = torch.max(probs, 1)  # B x N_seg
                    for i, prob in enumerate(probs):
                        idx = seg_point_list[i]
                        all_seg_logits[idx:idx + opt.seg_window_length] += prob.transpose(0, 1).cpu().numpy()

                end_idx = seg_point_list[-1] + opt.seg_window_length
                avg_logits = all_seg_logits / np.expand_dims(temporal_counts, axis=-1)
                avg_logits = avg_logits[:end_idx]  # N x K
                predictions = np.argmax(avg_logits, axis=-1)
                predictions = [opt.mapping_dict[str(key)] for key in predictions]

                if not os.path.exists(os.path.dirname(args.out_path)):
                    os.system('mkdir -p ' + os.path.dirname(args.out_path))
                with open(args.out_path, 'w') as f:  # 覆盖写
                    f.write("### Frame level recognition: ###\n")
                    f.write(' '.join(predictions))
                # reformat_pred(args.out_path, args.out_path)
                print("Frame seg:", frame_count)
            else:
                pass
                # print("Waiting for next frame ...")
                # print("Frame embed:", frame_count)
        else:
            break
    # ======================== Refreshing cache ========================
    embed_sess.close()


##ARGUMENTS: First Arg -- 1 for Training, 0 for Inference
##           Second Arg -- 1 for FrontView, 0 for Top View
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for Online Temporal Action Segmentation")
    parser.add_argument("--front_view", action="store_true",default=True)
    parser.add_argument("--top_view", action="store_true")
    parser.add_argument("--input_path", type=str,
                        default='/home/pingguo/wangbin/scale_balance_evaluation/video_seg/data/video_9_front.avi')
    parser.add_argument("--out_path", type=str, default='./seg_results.txt')
    args = parser.parse_args()
    main(args)
