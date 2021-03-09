"""
 Copyright (c) 2019 Intel Corporation

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

import cv2
import numpy as np


def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def postprocess(scores, classes, boxes, raw_cls_masks,
                im_h, im_w, im_scale_y=None, im_scale_x=None, im_scale=None,
                full_image_masks=True, encode_masks=False,
                confidence_threshold=0.0):
    no_detections = (np.empty((0, ), dtype=np.float32), np.empty((0, ), dtype=np.float32),\
                     np.empty((0, 4), dtype=np.float32), [])
    if scores is None:
        return no_detections

    scale = im_scale
    if scale is None:
        assert (im_scale_x is not None) and (im_scale_y is not None)
        scale = [im_scale_x, im_scale_y, im_scale_x, im_scale_y]

    confidence_filter = scores > confidence_threshold
    scores = scores[confidence_filter]
    classes = classes[confidence_filter]
    boxes = boxes[confidence_filter]
    if raw_cls_masks is not None:
        raw_cls_masks = list(segm for segm, is_valid in zip(raw_cls_masks, confidence_filter) if is_valid)

    if len(scores) == 0:
        return no_detections

    boxes = boxes / scale
    classes = classes.astype(np.uint32)
    if raw_cls_masks is not None:
        masks = []
        for box, cls, raw_mask in zip(boxes, classes, raw_cls_masks):
            raw_cls_mask = raw_mask[cls, ...]
            mask = segm_postprocess(box, raw_cls_mask, im_h, im_w, full_image_masks, encode_masks)
            masks.append(mask)
    else:
        masks = None

    return scores, classes, boxes, masks


def segm_postprocess(box, raw_cls_mask, im_h, im_w, full_image_mask=True, encode=False):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_boxes(box[np.newaxis, :],
                                raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0))[0]
    extended_box = extended_box.astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)

    if full_image_mask:
        # Put an object mask in an image mask.
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                                     (x0 - extended_box[0]):(x1 - extended_box[0])]
    else:
        original_box = box.astype(int)
        x0, y0 = np.clip(original_box[:2], a_min=0, a_max=[im_w, im_h])
        x1, y1 = np.clip(original_box[2:] + 1, a_min=0, a_max=[im_w, im_h])
        im_mask = np.ascontiguousarray(mask[(y0 - original_box[1]):(y1 - original_box[1]),
                                            (x0 - original_box[0]):(x1 - original_box[0])])
    return im_mask
