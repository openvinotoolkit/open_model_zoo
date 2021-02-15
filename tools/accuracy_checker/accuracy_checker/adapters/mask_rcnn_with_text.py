"""
Copyright (c) 2018-2021 Intel Corporation

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

from .mask_rcnn import MaskRCNNAdapter
from ..config import StringField, NumberField
from ..representation import TextDetectionPrediction


class MaskRCNNWithTextAdapter(MaskRCNNAdapter):
    __provider__ = 'mask_rcnn_with_text'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'classes_out': StringField(
                description="Name of output layer with information about classes.",
                optional=False
            ),
            'scores_out': StringField(
                description="Name of output layer with bbox scores.",
                optional=True
            ),
            'boxes_out': StringField(
                description="Name of output layer with bboxes.",
                optional=False
            ),
            'raw_masks_out': StringField(
                description='Name of output layer with raw instances masks.',
                optional=False
            ),
            'texts_out': StringField(
                description='Name of output layer with texts.',
                optional=False
            ),
            'confidence_threshold': NumberField(
                description='Confidence threshold that is used to filter out detected instances.',
                optional=False
            ),
        })

        return parameters

    def configure(self):
        self.classes_out = self.get_value_from_config('classes_out')
        self.scores_out = self.get_value_from_config('scores_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.num_detections_out = self.get_value_from_config('num_detections_out')
        self.raw_masks_out = self.get_value_from_config('raw_masks_out')
        self.texts_out = self.get_value_from_config('texts_out')
        self.confidence_threshold = self.get_value_from_config('confidence_threshold')

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)

        classes = raw_outputs[self.classes_out]
        if self.scores_out:
            valid_detections_mask = classes > 0
            scores = raw_outputs[self.scores_out][valid_detections_mask]
        else:
            scores = raw_outputs[self.boxes_out][:, 4]
            valid_detections_mask = scores > 0
            scores = scores[valid_detections_mask]
        classes = classes[valid_detections_mask].astype(np.uint32)
        boxes = raw_outputs[self.boxes_out][valid_detections_mask, :4]
        raw_masks = raw_outputs[self.raw_masks_out][valid_detections_mask]
        texts = raw_outputs[self.texts_out][valid_detections_mask]

        confidence_filter = scores > self.confidence_threshold
        classes = classes[confidence_filter]
        boxes = boxes[confidence_filter]
        texts = texts[confidence_filter]
        raw_masks = raw_masks[confidence_filter]

        text_filter = texts != ''
        classes = classes[text_filter]
        boxes = boxes[text_filter]
        texts = texts[text_filter]
        raw_masks = raw_masks[text_filter]

        results = []

        for identifier, image_meta in zip(identifiers, frame_meta):
            im_scale_x, im_scale_y = image_meta['scale_x'], image_meta['scale_y']
            img_h, img_w = image_meta['image_size'][:2]
            boxes[:, :4] /= np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
            boxes[:, 0:4:2] = np.clip(boxes[:, 0:4:2], 0, img_w - 1)
            boxes[:, 1:4:2] = np.clip(boxes[:, 1:4:2], 0, img_h - 1)

            segms = self.mask_to_result(
                boxes,
                classes,
                raw_masks,
                num_classes=1,
                mask_thr_binary=0.5,
                img_size=(img_h, img_w)
            )
            rectangles = self.masks_to_rects(segms[0])

            results.append(
                TextDetectionPrediction(identifier, points=rectangles, description=texts))

        return results

    @staticmethod
    def masks_to_rects(masks):
        rects = []
        for mask in masks:
            decoded_mask = mask.astype(np.uint8)
            contours = cv2.findContours(decoded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
            contour = sorted(contours, key=lambda x: -cv2.contourArea(x))[0]
            xys = cv2.boxPoints(cv2.minAreaRect(contour))
            rects.append(xys)

        return rects

    @staticmethod
    def mask_to_result(det_bboxes,
                       det_labels,
                       det_masks,
                       num_classes,
                       mask_thr_binary=0.5,
                       img_size=None):
        masks = det_masks
        bboxes = det_bboxes[:, :4]
        labels = det_labels

        cls_masks = [[] for _ in range(num_classes)]

        for bbox, label, mask in zip(bboxes, labels, masks):
            x0, y0, x1, y1 = bbox
            src_points = np.float32([[0, 0], [0, mask.shape[0]], [mask.shape[1],  mask.shape[0]]]) - 0.5
            dst_points = np.float32([[x0, y0], [x0, y1], [x1, y1]]) - 0.5
            transform_matrix = cv2.getAffineTransform(src_points, dst_points)
            mask = cv2.warpAffine(mask, transform_matrix, img_size[::-1]) 
            mask = (mask >= mask_thr_binary).astype(np.uint8)
            cls_masks[label].append(mask)

        return cls_masks
