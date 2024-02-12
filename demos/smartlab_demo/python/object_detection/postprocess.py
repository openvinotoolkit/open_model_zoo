"""
 Copyright (C) 2021-2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the Lic ense for the specific language governing permissions and
 limitations under the License.
"""

import cv2
import numpy as np
from .deploy_util import nms, multiclass_nms


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # original source: https://github.com/Megvii-BaseDetection/YOLOX
    # Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

    box_corner = np.zeros(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if len(image_pred[0]) == 0:
            continue
        # Get score and class with highest confidence
        class_conf = np.max(image_pred[:, 5:5+num_classes], axis=1, keepdims=True)
        class_pred = np.argmax(image_pred[: , 5:5+num_classes], axis=1)
        class_pred = np.expand_dims(class_pred, axis=1)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = np.concatenate((image_pred[:, :5], class_conf, class_pred), axis=1)
        detections = detections[conf_mask]

        if len(detections)<1:
            continue
        if class_agnostic:
            nms_out_index = nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre)
        else:
            nms_out_index = multiclass_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre)

        detections = detections[nms_out_index] # filtering boxes

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = np.concatenate((output[i], detections))
    return output

def crop_pad_resize(img: np.ndarray, roi_xyxy: np.ndarray, dsize):
    assert img.ndim == 3 and roi_xyxy.ndim == 1
    assert roi_xyxy.shape[0] == 4

    ### crop image
    rx0, ry0, rx1, ry1 = roi_xyxy.astype(int)
    img_roi = img[ry0: ry1, rx0: rx1]

    ### pad image
    hs, ws = img_roi.shape[:2]
    if ws < hs: # taller image
        padh1 = padh2 = 0
        padw1 = (hs - ws) // 2
        padw2 = hs - (ws + padw1)
        img_res = np.pad(img_roi, ([padh1, padh2], [padw1, padw2], [0, 0]))
    elif ws > hs: # wider image
        padw1 = padw2 = 0
        padh1 = (ws - hs) // 2
        padh2 = ws - (hs + padh1)
        img_res = np.pad(img_roi, ([padh1, padh2], [padw1, padw2], [0, 0]))
    else: # 4 edges identical
        padh1 = padh2 = 0
        padw1 = padw2 = 0
        img_res = img_roi
    hp, wp = img_res.shape[:2]

    ### resize image
    if dsize != hp or dsize != wp:
        img_res = cv2.resize(img_res, (dsize, dsize))

    ### organize & output result
    pad_img_size = (hp, wp)
    pad_left = padw1
    pad_top = padh1

    return img_res, pad_img_size, pad_left, pad_top

def reverse_crop_pad_resize(
    xyxys: np.ndarray,
    pad_img_size: tuple,
    dsize: int,
    pad_left: int,
    pad_top: int,
    roi_xyxy: np.ndarray):
    '''
        reversal of crop_pad_resize
        xyxys: bboxes in xyxy format
        pad_img_size, pad_left, pad_top are from outputs of crop_pad_resize()
        dsize is the resize params used in crop_pad_resize()
        roi_xyxy is the ROI used to crop original image
    '''
    ## resize & un-pad bboxes back to padded image
    hp, wp = pad_img_size
    scalex, scaley = dsize / wp, dsize / hp
    xyxys[:, 0: : 2] = np.clip(xyxys[:, 0: : 2] / scalex - pad_left, 0, wp)
    xyxys[:, 1: : 2] = np.clip(xyxys[:, 1: : 2] / scaley - pad_top, 0, hp)
    ##  un-crop
    offsetx, offsety = roi_xyxy[: 2]
    xyxys[:, 0: : 2] += offsetx
    xyxys[:, 1: : 2] += offsety

    return xyxys
