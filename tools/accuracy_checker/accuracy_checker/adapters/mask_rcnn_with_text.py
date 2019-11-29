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

import numpy as np
import cv2

from .mask_rcnn import MaskRCNNAdapter
from ..config import StringField, ConfigError, NumberField
from ..representation import ContainerPrediction, TextDetectionPrediction
from ..utils import contains_all


class MaskRCNNWithTextAdapter(MaskRCNNAdapter):
    __provider__ = 'mask_rcnn_with_text'

    def __init__(self, launcher_config, label_map=None, output_blob=None):
        super().__init__(launcher_config, label_map, output_blob)


    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'classes_out': StringField(
                description="Name of output layer with information about classes. "
                            "(optional, if your model has detection_output layer as output).",
                optional=True
            ),
            'scores_out': StringField(
                description="Name of output layer with bbox scores."
                            "(optional, if your model has detection_output layer as output).",
                optional=True
            ),
            'boxes_out': StringField(
                description="Name of output layer with bboxes."
                            "(optional, if your model has detection_output layer as output).",
                optional=True
            ),
            'raw_masks_out': StringField(
                description='Name of output layer with raw instances masks'
            ),
            'texts_out': StringField(
                description='Name of output layer with texts'
            ),
            'confidence_threshold': NumberField(
                description='Confidence threshold that is used to filter out detected instances'
            ),
            'num_detections_out': StringField(
                optional=True, description='Name of output layer with number valid detections '
                                           '(used in MaskRCNN models trained with TF Object Detection API)'
            ),
            'detection_out': StringField(
                description='SSD-like detection output layer name '
                            '(optional, if your model has scores_out, boxes_out and classes_out).',
                optional=True
            )
        })

        return parameters

    def configure(self):
        box_outputs = ['classes_out', 'scores_out', 'boxes_out']
        detection_out = 'detection_out'
        if contains_all(self.launcher_config, [*box_outputs, detection_out]):
            raise ConfigError('only detection output or [{}] should be provided'.format(', '.join(box_outputs)))
        self.detection_out = self.get_value_from_config(detection_out)
        if not self.detection_out:
            if not contains_all(self.launcher_config, box_outputs):
                raise ConfigError('all related outputs should be specified: {}'.format(', '.join(box_outputs)))
            self.classes_out = self.get_value_from_config('classes_out')
            self.scores_out = self.get_value_from_config('scores_out')
            self.boxes_out = self.get_value_from_config('boxes_out')
            self.num_detections_out = self.get_value_from_config('num_detections_out')

        self.raw_masks_out = self.get_value_from_config('raw_masks_out')
        self.texts_out = self.get_value_from_config('texts_out')
        if self.detection_out:
            raise NotImplementedError

        if self.num_detections_out:
            raise NotImplementedError

        self.confidence_threshold = self.get_value_from_config('confidence_threshold')

        self.realisation = self._process_pytorch_outputs

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        return self.realisation(raw_outputs, identifiers, frame_meta)

    def _process_pytorch_outputs(self, raw_outputs, identifiers, frame_meta):
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
        # raw_masks = list(
        #     segm for segm, is_valid in zip(raw_masks, confidence_filter) if is_valid)

        results = []

        for identifier, image_meta in zip(identifiers, frame_meta):
            original_image_size = image_meta['image_size'][:2]
            if 'scale_x' in image_meta and 'scale_y' in image_meta:
                im_scale_x = image_meta['scale_x']
                im_scale_y = image_meta['scale_y']
            else:
                processed_image_size = next(image_meta['input_shape'])[1:]
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

            results.append(TextDetectionPrediction(identifier, points=rectangles, description=texts))

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
