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

from collections import namedtuple
import cv2
import numpy as np
from models.utils import nms

ModelAttributes = namedtuple('ModelAttributes', ['required_outputs', 'postprocessor'])


def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                            (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask


def mask_rcnn_postprocess(
        outputs, scale_x, scale_y, frame_height, frame_width, input_height, input_width, conf_threshold
):
    segmentoly_postprocess = 'raw_masks' in outputs
    boxes = outputs['boxes'] if segmentoly_postprocess else outputs['boxes'][:, :4]
    scores = outputs['scores'] if segmentoly_postprocess else outputs['boxes'][:, 4]
    boxes[:, 0::2] /= scale_x
    boxes[:, 1::2] /= scale_y
    if segmentoly_postprocess:
        classes = outputs['classes'].astype(np.uint32)
    else:
        classes = outputs['labels'].astype(np.uint32) + 1
    masks = []
    masks_name = 'raw_masks' if segmentoly_postprocess else 'masks'
    for box, cls, raw_mask in zip(boxes, classes, outputs[masks_name]):
        raw_cls_mask = raw_mask[cls, ...] if segmentoly_postprocess else raw_mask
        mask = segm_postprocess(box, raw_cls_mask, frame_height, frame_width)
        masks.append(mask)
    # Filter out detections with low confidence.
    detections_filter = scores > conf_threshold
    scores = scores[detections_filter]
    classes = classes[detections_filter]
    boxes = boxes[detections_filter]
    masks = [segm for segm, is_valid in zip(masks, detections_filter) if is_valid]
    return scores, classes, boxes, masks


def yolact_postprocess(
        outputs, scale_x, scale_y, frame_height, frame_width, input_height, input_width, conf_threshold
):
    boxes = outputs['boxes'][0]
    conf = np.transpose(outputs['conf'][0])
    masks = outputs['mask'][0]
    proto = outputs['proto'][0]
    num_classes = conf.shape[0]
    idx_lst, cls_lst, scr_lst = [], [], []
    shift_x = (input_width - (frame_width * scale_x)) / frame_width
    shift_y = (input_height - (frame_height * scale_y)) / frame_height

    for cls in range(1, num_classes):
        cls_scores = conf[cls, :]
        idx = np.arange(cls_scores.shape[0])
        conf_mask = cls_scores > conf_threshold

        cls_scores = cls_scores[conf_mask]
        idx = idx[conf_mask]

        if cls_scores.shape[0] == 0:
            continue
        x1, x2 = sanitize_coordinates(boxes[idx, 0], boxes[idx, 2], frame_width)
        y1, y2 = sanitize_coordinates(boxes[idx, 1], boxes[idx, 3], frame_height)
        keep = nms(x1, y1, x2, y2, cls_scores, 0.5)

        idx_lst.append(idx[keep])
        cls_lst.append(np.full(len(keep), cls))
        scr_lst.append(cls_scores[keep])

    if not idx_lst:
        return np.array([]), np.array([]), np.array([]), np.array([])
    idx = np.concatenate(idx_lst, axis=0)
    classes = np.concatenate(cls_lst, axis=0)
    scores = np.concatenate(scr_lst, axis=0)

    idx2 = np.argsort(scores, axis=0)[::-1]
    scores = scores[idx2]

    idx = idx[idx2]
    classes = classes[idx2]

    boxes = boxes[idx]
    masks = masks[idx]
    if np.size(boxes) > 0:
        boxes, scores, classes, masks = yolact_segm_postprocess(
            boxes, masks, scores, classes, proto, frame_width, frame_height, shift_x=shift_x, shift_y=shift_y
        )
    return scores, classes, boxes, masks


def sanitize_coordinates(_x1, _x2, img_size, shift=0, padding=0):
    _x1 = (_x1 + shift / 2) * img_size
    _x2 = (_x2 + shift / 2) * img_size
    x1 = np.clip(_x1 - padding, 0, img_size)
    x2 = np.clip(_x2 + padding, 0, img_size)

    return x1, x2


def yolact_segm_postprocess(
        boxes, masks, score, classes, proto_data, w, h,
        score_threshold=0, shift_x=0, shift_y=0):
    def crop_mask(masks, boxes, padding: int = 1):
        h, w, n = np.shape(masks)
        x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding=padding)
        y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding=padding)

        rows = np.reshape(
            np.repeat(np.reshape(np.repeat(np.arange(w, dtype=x1.dtype), h), (w, h)), n, axis=-1), (h, w, n)
        )
        cols = np.reshape(
            np.repeat(np.reshape(np.repeat(np.arange(h, dtype=x1.dtype), h), (w, h)), n, axis=-1), (h, w, n)
        )
        rows = np.transpose(rows, (1, 0, 2))

        masks_left = rows >= x1
        masks_right = rows < x2
        masks_up = cols >= y1
        masks_down = cols < y2

        crop_mask = masks_left * masks_right * masks_up * masks_down
        return masks * crop_mask

    if score_threshold > 0:
        keep = score > score_threshold
        score = score[keep]
        boxes = boxes[keep]
        masks = masks[keep]
        classes = classes[keep]

        if np.size(score) == 0:
            return [] * 4

    masks = proto_data @ masks.T
    masks = 1 / (1 + np.exp(-masks))
    masks = crop_mask(masks, boxes)

    masks = np.transpose(masks, (2, 0, 1))
    x1, x2 = np.ceil(sanitize_coordinates(boxes[:, 0], boxes[:, 2], w)).astype(int)
    y1, y2 = np.ceil(sanitize_coordinates(boxes[:, 1], boxes[:, 3], h)).astype(int)
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, shift_x)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, shift_y)
    ready_masks = []

    for mask in masks:
        mask = cv2.resize(mask, (w, h), cv2.INTER_LINEAR)
        mask = mask > 0.5
        ready_masks.append(mask.astype(np.uint8))

    return boxes, score, classes, ready_masks


MODEL_ATTRIBUTES = {
    'mask_rcnn_segmentoly': ModelAttributes(('boxes', 'scores', 'classes', 'raw_masks'),
                                            mask_rcnn_postprocess),
    'mask_rcnn': ModelAttributes(('boxes', 'labels', 'masks'), mask_rcnn_postprocess),
    'yolact': ModelAttributes(('boxes', 'conf', 'proto', 'mask'), yolact_postprocess)
}


def check_model(net):
    num_inputs = len(net.input_info)
    assert num_inputs <= 2, 'Demo supports only topologies with 1 or 2 inputs.'
    image_input = [input_name for input_name, in_info in net.input_info.items() if len(in_info.input_data.shape) == 4]
    assert len(image_input) == 1, 'Demo supports only model with single input for images'
    image_input = image_input[0]
    image_info_input = None
    if num_inputs == 2:
        image_info_input = [
            input_name for input_name, in_info in net.input_info.items()
            if len(in_info.input_data.shape) == 2 and in_info.input_data.shape[-1] == 3
        ]
        assert len(image_info_input) == 1, 'Demo supports only model with single im_info input'
        image_info_input = image_info_input[0]
    if image_info_input:
        model_type = 'mask_rcnn_segmentoly'
    else:
        model_type = 'mask_rcnn' if len(net.input_info) == 1 and len(net.outputs) in [3, 5] else 'yolact'
    model_attributes = MODEL_ATTRIBUTES[model_type]
    assert set(model_attributes.required_outputs) <= net.outputs.keys(), \
        'Demo supports only topologies with the following output keys: {}'.format(
            ', '.join(model_attributes.required_outputs))

    input_shape = net.input_info[image_input].input_data.shape
    assert input_shape[0] == 1, 'Only batch 1 is supported by the demo application'

    return image_input, image_info_input, input_shape, model_type, model_attributes.postprocessor
