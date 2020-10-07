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
                optional=False
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
        valid_detections_mask = classes > 0
        classes = classes[valid_detections_mask]
        boxes = raw_outputs[self.boxes_out][valid_detections_mask]
        scores = raw_outputs[self.scores_out][valid_detections_mask]
        raw_masks = raw_outputs[self.raw_masks_out][valid_detections_mask]
        texts = raw_outputs[self.texts_out][valid_detections_mask]

        confidence_filter = scores > self.confidence_threshold
        classes = classes[confidence_filter]
        boxes = boxes[confidence_filter]
        texts = texts[confidence_filter]
        raw_masks = raw_masks[confidence_filter]

        results = []

        for identifier, image_meta in zip(identifiers, frame_meta):
            original_image_size = image_meta['image_size'][:2]
            if 'scale_x' in image_meta and 'scale_y' in image_meta:
                im_scale_x = image_meta['scale_x']
                im_scale_y = image_meta['scale_y']
            else:
                image_input = [shape for shape in image_meta['input_shape'].values() if len(shape) == 4]
                assert image_input, "image input not found"
                assert len(image_input) == 1, 'several input images detected'
                processed_image_size = image_input[0][2:]
                im_scale_y = processed_image_size[0] / original_image_size[0]
                im_scale_x = processed_image_size[1] / original_image_size[1]
            boxes[:, 0::2] /= im_scale_x
            boxes[:, 1::2] /= im_scale_y
            classes = classes.astype(np.uint32)
            masks = []
            raw_mask_for_all_classes = np.shape(raw_masks)[1] != len(identifiers)
            if raw_mask_for_all_classes:
                per_obj_raw_masks = []
                for cls, raw_mask in zip(classes, raw_masks):
                    per_obj_raw_masks.append(raw_mask[cls, ...])
            else:
                per_obj_raw_masks = np.squeeze(raw_masks, axis=1)

            for box, raw_cls_mask in zip(boxes, per_obj_raw_masks):
                mask = self.segm_postprocess(box, raw_cls_mask, *original_image_size, True, False)
                masks.append(mask)

            rectangles = self.masks_to_rects(masks)

            results.append(
                TextDetectionPrediction(identifier, points=rectangles, description=texts))

        return results

    @staticmethod
    def masks_to_rects(masks):
        rects = []
        for mask in masks:
            decoded_mask = mask
            contours = cv2.findContours(decoded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

            areas = []
            boxes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                areas.append(area)

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                boxes.append(box)

            if areas:
                i = np.argmax(areas)
                rects.append(boxes[i])

        return rects
