"""
Copyright (c) 2018-2024 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import OrderedDict
import cv2
import numpy as np

from ...adapters import MTCNNPAdapter


def calibrate_predictions(previous_stage_predictions, out, threshold, outputs_mapping, iou_type=None):
    prob_out = outputs_mapping['probability_out']
    if prob_out not in out[0]:
        prob_out = prob_out + '/sink_port_0' if '/sink_port_0' not in prob_out else prob_out.replace('/sink_port_0', '')
    score = out[0][prob_out][:, 1]
    pass_t = np.where(score > 0.7)[0]
    removed_boxes = [i for i in range(previous_stage_predictions[0].size) if i not in pass_t]
    previous_stage_predictions[0].remove(removed_boxes)
    previous_stage_predictions[0].scores = score[pass_t]
    bboxes = np.c_[
        previous_stage_predictions[0].x_mins, previous_stage_predictions[0].y_mins,
        previous_stage_predictions[0].x_maxs, previous_stage_predictions[0].y_maxs,
        previous_stage_predictions[0].scores
    ]
    region_out = outputs_mapping['region_out']
    if region_out not in out[0]:
        region_out = (
            region_out + '/sink_port_0' if '/sink_port_0' not in region_out else region_out.replace('/sink_port_0', '')
        )
    mv = out[0][region_out][pass_t]
    if iou_type:
        previous_stage_predictions[0], peek = nms(previous_stage_predictions[0], threshold, iou_type)
        bboxes = np.c_[
            previous_stage_predictions[0].x_mins, previous_stage_predictions[0].y_mins,
            previous_stage_predictions[0].x_maxs, previous_stage_predictions[0].y_maxs,
            previous_stage_predictions[0].scores
        ]
        mv = mv[np.sort(peek).astype(int)]
    x_mins, y_mins, x_maxs, y_maxs, _ = bbreg(bboxes, mv.T).T
    previous_stage_predictions[0].x_mins = x_mins
    previous_stage_predictions[0].y_mins = y_mins
    previous_stage_predictions[0].x_maxs = x_maxs
    previous_stage_predictions[0].y_maxs = y_maxs
    return previous_stage_predictions


def nms(prediction, threshold, iou_type):
    bboxes = np.c_[prediction.x_mins, prediction.y_mins, prediction.x_maxs, prediction.y_maxs, prediction.scores]
    peek = MTCNNPAdapter.nms(bboxes, threshold, iou_type)
    prediction.remove([i for i in range(prediction.size) if i not in peek])
    return prediction, peek


def bbreg(boundingbox, reg):
    reg = reg.T
    # calibrate bounding boxes
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    bb0 = boundingbox[:, 0] + reg[:, 0] * w
    bb1 = boundingbox[:, 1] + reg[:, 1] * h
    bb2 = boundingbox[:, 2] + reg[:, 2] * w
    bb3 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.array([bb0, bb1, bb2, bb3]).T
    return boundingbox


def filter_valid(dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph):
    mask = np.ones(len(tmph))
    tmp_ys_len = (edy + 1) - dy
    tmp_xs_len = (edx + 1) - dx
    img_ys_len = (ey + 1) - y
    img_xs_len = (ex + 1) - x
    mask = np.logical_and(mask, np.logical_and(tmph > 0, tmpw > 0))
    mask = np.logical_and(mask, np.logical_and(tmp_ys_len > 0, tmp_xs_len > 0))
    mask = np.logical_and(mask, np.logical_and(img_xs_len > 0, img_ys_len > 0))
    mask = np.logical_and(mask, np.logical_and(tmp_xs_len == img_xs_len, tmp_ys_len == img_ys_len))
    return dy[mask], edy[mask], dx[mask], edx[mask], y[mask], ey[mask], x[mask], ex[mask], tmpw[mask], tmph[mask], mask


def pad(boxesA, h, w):
    boxes = boxesA.copy()
    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1
    numbox = boxes.shape[0]
    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw
    edy = tmph
    x = boxes[:, 0:1][:, 0]
    y = boxes[:, 1:2][:, 0]
    ex = boxes[:, 2:3][:, 0]
    ey = boxes[:, 3:4][:, 0]
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w - 1 + tmpw[tmp]
        ex[tmp] = w - 1
    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h - 1 + tmph[tmp]
        ey[tmp] = h - 1
    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])
    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    # for python index from 0, while matlab from 1
    dy, dx = np.maximum(0, dy - 1), np.maximum(0, dx - 1)
    y = np.maximum(0, y - 1)
    x = np.maximum(0, x - 1)
    edy = np.maximum(0, edy - 1)
    edx = np.maximum(0, edx - 1)
    ey = np.maximum(0, ey - 1)
    ex = np.maximum(0, ex - 1)
    return filter_valid(dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph)


def rerec(bboxA):
    w = bboxA[:, 2] - bboxA[:, 0]
    h = bboxA[:, 3] - bboxA[:, 1]
    max_side = np.maximum(w, h).T
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - max_side * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - max_side * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.repeat([max_side], 2, axis=0).T
    return bboxA


def cut_roi(image, prediction, dst_size, include_bound=True):
    bboxes = np.c_[prediction.x_mins, prediction.y_mins, prediction.x_maxs, prediction.y_maxs, prediction.scores]
    img = image.data
    bboxes = rerec(bboxes)
    bboxes[:, 0:4] = np.fix(bboxes[:, 0:4])
    dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph, mask = pad(bboxes, *img.shape[:2])
    bboxes = bboxes[mask]
    numbox = bboxes.shape[0]
    tempimg = np.zeros((numbox, dst_size, dst_size, 3))
    for k in range(numbox):
        tmp_k_h, tmp_k_w = int(tmph[k]) + int(include_bound), int(tmpw[k]) + int(include_bound)
        tmp = np.zeros((tmp_k_h, tmp_k_w, 3))
        tmp_ys, tmp_xs = slice(int(dy[k]), int(edy[k]) + 1), slice(int(dx[k]), int(edx[k]) + 1)
        img_ys, img_xs = slice(int(y[k]), int(ey[k]) + 1), slice(int(x[k]), int(ex[k]) + 1)
        tmp[tmp_ys, tmp_xs] = img[img_ys, img_xs]
        tempimg[k, :, :, :] = cv2.resize(tmp, (dst_size, dst_size))
    image.data = tempimg
    return image


def transform_for_callback(batch_size, raw_outputs):
    output_per_box = []
    fq_weights = []
    for i in range(batch_size):
        box_outs = OrderedDict()
        for layer_node, data in raw_outputs[0].items():
            if layer_node in fq_weights:
                continue
            if layer_node.get_node().friendly_name.endswith('fq_weights_1'):
                fq_weights.append(layer_node)
                box_outs[layer_node] = data
            elif data.ndim == 0:
                box_outs[layer_node] = np.array([data])
            elif data.shape[0] <= i:
                box_outs[layer_node] = data
            else:
                box_outs[layer_node] = np.expand_dims(data[i], axis=0)
        output_per_box.append(box_outs)
    return output_per_box
