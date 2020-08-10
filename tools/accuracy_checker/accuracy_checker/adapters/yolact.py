"""
Copyright (c) 2018-2020 Intel Corporation

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

try:
    import pycocotools.mask as mask_util
except ImportError:
    mask_util = None

from ..config import StringField, NumberField
from ..postprocessor import NMS
from ..representation import DetectionPrediction, CoCocInstanceSegmentationPrediction, ContainerPrediction

from .adapter import Adapter


class YolactAdapter(Adapter):
    __provider__ = 'yolact'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'loc_out': StringField(description='name of output with box locations'),
            'conf_out': StringField(description='name of output with confidence scores'),
            'prior_out': StringField(description='name of output with prior boxes'),
            'mask_out': StringField(description='name of output with masks'),
            'proto_out': StringField(description='name of output with proto for masks calculation'),
            'confidence_threshold':  NumberField(
                value_type=float, optional=True, default=0.05, description='confidence threshold'
            ),
            'max_detections': NumberField(
                value_type=int, optional=True, default=100, description='max number of detections'
            )
        })
        return params

    def configure(self):
        if mask_util is None:
            raise ImportError('pycocotools is not installed. Please install it before using mask_rcnn adapter.')
        self.encoder = mask_util.encode
        self.loc_out = self.get_value_from_config('loc_out')
        self.conf_out = self.get_value_from_config('conf_out')
        self.prior_out = self.get_value_from_config('prior_out')
        self.mask_out = self.get_value_from_config('mask_out')
        self.proto_out = self.get_value_from_config('proto_out')
        self.conf_thresh = self.get_value_from_config('confidence_threshold')
        self.max_num_detections = self.get_value_from_config('max_detections')

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        prior_boxes = raw_outputs[self.prior_out]
        result = []
        for identifier, locs, conf, masks, proto, meta in zip(
                identifiers, raw_outputs[self.loc_out], raw_outputs[self.conf_out],
                raw_outputs[self.mask_out], raw_outputs[self.proto_out], frame_meta
        ):
            h, w, _ = meta['image_size']
            boxes = self.decode_boxes(locs, prior_boxes)
            conf = np.transpose(conf)
            cur_scores = conf[1:, :]
            conf_scores = np.max(cur_scores, axis=0)

            keep = (conf_scores > self.conf_thresh)
            scores = cur_scores[:, keep]
            boxes = boxes[keep, :]
            masks = masks[keep, :]

            if scores.shape[1] == 0:
                return [ContainerPrediction(
                    {'detection_prediction': DetectionPrediction(identifier, [], [], [], [], [], []),
                     'segmentation_prediction': CoCocInstanceSegmentationPrediction(identifier, [], [], [])}
                )]
            num_classes = scores.shape[0]
            idx_lst, cls_lst, scr_lst = [], [], []

            for _cls in range(num_classes):
                cls_scores = scores[_cls, :]
                conf_mask = cls_scores > self.conf_thresh
                idx = np.arange(cls_scores.shape[0])

                cls_scores = cls_scores[conf_mask]
                idx = idx[conf_mask]

                if cls_scores.shape[0] == 0:
                    continue
                keep = NMS.nms(*boxes[conf_mask].T, cls_scores, 0.5, include_boundaries=False)

                idx_lst.append(idx[keep])
                cls_lst.append(np.full(len(keep), _cls))
                scr_lst.append(cls_scores[keep])

            idx = np.concatenate(idx_lst, axis=0)
            classes = np.concatenate(cls_lst, axis=0)
            scores = np.concatenate(scr_lst, axis=0)

            idx2 = np.argsort(scores, axis=0)[::-1]
            scores = scores[idx2]
            idx2 = idx2[:self.max_num_detections]
            scores = scores[:self.max_num_detections]

            idx = idx[idx2]
            classes = classes[idx2]

            boxes = boxes[idx]
            masks = masks[idx]
            if np.size(boxes) > 0:
                boxes, scores, classes, masks = self.postprocess(boxes, masks, scores, classes, proto, w, h)
            if np.size(boxes):
                x_mins, y_mins, x_maxs, y_maxs = boxes.T
            else:
                x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
            det_prediction = DetectionPrediction(identifier, classes, scores, x_mins, y_mins, x_maxs, y_maxs)
            segm_prediction = CoCocInstanceSegmentationPrediction(identifier, masks, classes, scores)
            segm_prediction.metadata['image_size'] = meta['image_size']
            result.append(ContainerPrediction(
                {'detection_prediction': det_prediction, 'segmentation_prediction': segm_prediction}
            ))

        return result

    def postprocess(self, boxes, masks, score, classes, proto_data, w, h, crop_masks=True, score_threshold=0):
        if score_threshold > 0:
            keep = score > score_threshold
            score = score[keep]
            boxes = boxes[keep]
            masks = masks[keep]
            classes = classes[keep]

            if np.size(score) == 0:
                return [] * 4

        masks = proto_data @ masks.T
        masks = self.mask_proto_mask_activation(masks)

        # Crop masks before upsampling because you know why
        if crop_masks:
            masks = self.crop_mask(masks, boxes)

            # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = np.transpose(masks, (2, 0, 1))
        ready_masks = []

        # Scale masks up to the full image
        for mask in masks:
            mask = cv2.resize(mask, (w, h), cv2.INTER_LINEAR)
            mask = mask > 0.5
            im_mask = self.encoder(np.array(mask[:, :, np.newaxis].astype(np.uint8), order='F'))[0]
            im_mask['counts'] = im_mask['counts'].decode('utf-8')
            ready_masks.append(im_mask)
        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w)
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h)

        return boxes, score, classes, ready_masks

    @staticmethod
    def decode_boxes(loc, priors):
        variances = [0.1, 0.2]

        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes

    @staticmethod
    def mask_proto_mask_activation(masks):
        return 1 / (1 + np.exp(-masks))

    @staticmethod
    def crop_mask(masks, boxes, padding: int = 1):
        h, w, n = np.shape(masks)
        x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
        y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

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


def sanitize_coordinates(_x1, _x2, img_size, padding=0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    x1 = np.clip(_x1 - padding, 0, img_size)
    x2 = np.clip(_x2 + padding, 0, img_size)

    return x1, x2
