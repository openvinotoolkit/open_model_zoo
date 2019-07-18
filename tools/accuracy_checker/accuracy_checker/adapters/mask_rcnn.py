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
from .adapter import Adapter
from ..config import ConfigValidator, StringField, ConfigError
from ..representation import CoCocInstanceSegmentationPrediction, DetectionPrediction, ContainerPrediction
from ..utils import contains_all

class MaskRCNNAdapterConfig(ConfigValidator):
    type = StringField()
    classes_out = StringField(optional=True)
    scores_out = StringField(optional=True)
    boxes_out = StringField(optional=True)
    detection_out = StringField(optional=True)
    raw_masks_out = StringField()


class MaskRCNNAdapter(Adapter):
    __provider__ = 'mask_rcnn'

    def __init__(self, launcher_config, label_map=None, output_blob=None):
        super().__init__(launcher_config, label_map, output_blob)
        try:
            import pycocotools.mask as mask_util
            self.encoder = mask_util.encode
        except ImportError:
            raise ImportError('pycocotools is not installed. Please install it before using mask_rcnn adapter.')

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
            'detection_out': StringField(
                description='SSD-like detection output layer name '
                            '(optional, if your model has scores_out, boxes_out and classes_out).',
                optional=True
            )
        })

        return parameters

    def validate_config(self):
        config_valiator = MaskRCNNAdapterConfig(
            'MaskRCNN_adapter_config', MaskRCNNAdapterConfig.ERROR_ON_EXTRA_ARGUMENT
        )
        config_valiator.validate(self.launcher_config)

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

        self.raw_masks_out = self.get_value_from_config('raw_masks_out')

    def process(self, raw, identifiers=None, frame_meta=None):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.detection_out:
            return self._process_original_outputs(raw_outputs, identifiers, frame_meta)
        return self._process_detection_output(raw_outputs, identifiers, frame_meta)

    def _process_original_outputs(self, raw_outputs, identifiers, frame_meta):
        classes = raw_outputs[self.classes_out]
        valid_detections_mask = classes > 0
        classes = classes[valid_detections_mask]
        boxes = raw_outputs[self.boxes_out][valid_detections_mask]
        scores = raw_outputs[self.scores_out][valid_detections_mask]
        raw_masks = raw_outputs[self.raw_masks_out][valid_detections_mask]

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
            for box, cls, raw_mask in zip(boxes, classes, raw_masks):
                raw_cls_mask = raw_mask[cls, ...]
                mask = self.segm_postprocess(box, raw_cls_mask, *original_image_size, True, True)
                masks.append(mask)
            x_mins, y_mins, x_maxs, y_maxs = boxes.T
            detection_prediction = DetectionPrediction(identifier, classes, scores, x_mins, y_mins, x_maxs, y_maxs)
            instance_segmentation_prediction = CoCocInstanceSegmentationPrediction(identifier, masks, classes, scores)
            instance_segmentation_prediction.metadata['rects'] = np.c_[x_mins, y_mins, x_maxs, y_maxs]
            instance_segmentation_prediction.metadata['image_size'] = image_meta['image_size']
            results.append(ContainerPrediction({
                'detection_prediction': detection_prediction,
                'segmentation_prediction': instance_segmentation_prediction
            }))

            return results

    def _process_detection_output(self, raw_outputs, identifiers, frame_meta):
        raw_masks = raw_outputs[self.raw_masks_out]
        detections_boxes = raw_outputs[self.detection_out]

        results = []
        empty_box_index = -1
        empty_boxes_position = np.where(detections_boxes[:, 0] == empty_box_index)[0]
        if empty_boxes_position.size:
            detections_boxes = detections_boxes[:empty_boxes_position[0]]
            raw_masks = raw_masks[:empty_boxes_position[0]]

        for batch_index, identifier in enumerate(identifiers):
            image_size = frame_meta[batch_index]['image_size'][:2]
            prediction_box_mask = np.where(detections_boxes[:, 0] == batch_index)
            filtered_detections_boxes = detections_boxes[prediction_box_mask]
            filtered_detections_boxes = filtered_detections_boxes[:, 1::]
            filtered_masks = raw_masks[prediction_box_mask]
            detection_prediction = DetectionPrediction(identifier, *zip(*filtered_detections_boxes))
            instance_masks = []
            for box, masks in zip(filtered_detections_boxes, filtered_masks):
                label = box[0]
                cls_mask = masks[int(label)-1, ...]
                box[2::2] *= image_size[1]
                box[3::2] *= image_size[0]
                cls_mask = self.segm_postprocess(box[2:], cls_mask, *image_size, True, True)
                instance_masks.append(cls_mask)
            instance_segmentation_prediction = CoCocInstanceSegmentationPrediction(
                identifier, instance_masks, detection_prediction.labels, detection_prediction.scores
            )
            instance_segmentation_prediction.metadata['image_size'] = frame_meta[batch_index]['image_size']
            results.append(ContainerPrediction({
                'detection_prediction': detection_prediction,
                'segmentation_prediction': instance_segmentation_prediction
            }))

        return results

    def segm_postprocess(self, box, raw_cls_mask, im_h, im_w, full_image_mask=False, encode=False):
        # Add zero border to prevent upsampling artifacts on segment borders.
        raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
        extended_box = self.expand_boxes(box[np.newaxis, :], raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0))[0]
        extended_box = extended_box.astype(int)
        w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
        x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
        x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

        raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
        mask = raw_cls_mask.astype(np.uint8)

        if full_image_mask:
            # Put an object mask in an image mask.
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
            im_mask[y0:y1, x0:x1] = mask[
                (y0 - extended_box[1]):(y1 - extended_box[1]),
                (x0 - extended_box[0]):(x1 - extended_box[0])
            ]
        else:
            original_box = box.astype(int)
            x0, y0 = np.clip(original_box[:2], a_min=0, a_max=[im_w, im_h])
            x1, y1 = np.clip(original_box[2:] + 1, a_min=0, a_max=[im_w, im_h])
            im_mask = np.ascontiguousarray(
                mask[(y0 - original_box[1]):(y1 - original_box[1]), (x0 - original_box[0]):(x1 - original_box[0])]
            )

        if encode:
            im_mask = self.encoder(np.array(im_mask[:, :, np.newaxis].astype(np.uint8), order='F'))[0]
            im_mask['counts'] = im_mask['counts'].decode('utf-8')

        return im_mask

    @staticmethod
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
