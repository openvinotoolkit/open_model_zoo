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

import warnings

import cv2
import numpy as np

from .adapter import Adapter
from ..config import StringField, ConfigError
from ..representation import CoCoInstanceSegmentationPrediction, DetectionPrediction, ContainerPrediction
from ..postprocessor import FRCNNPostprocessingBboxResize
from ..utils import UnsupportedPackage

try:
    import pycocotools.mask as mask_util
except ImportError as import_error:
    mask_util = UnsupportedPackage("pycocotools", import_error.msg)


class MaskRCNNAdapter(Adapter):
    __provider__ = 'mask_rcnn'

    def __init__(self, launcher_config, label_map=None, output_blob=None):
        super().__init__(launcher_config, label_map, output_blob)
        if isinstance(mask_util, UnsupportedPackage):
            mask_util.raise_error(self.__provider__)
        self.encoder = mask_util.encode

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
                description='Name of output layer with raw instances masks',
                optional=True
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
        def is_detection_out(config):
            return bool(config.get('detection_out'))

        def is_box_outputs(config, box_outputs):
            for elem in box_outputs:
                if not config.get(elem):
                    return False

            return True

        box_outputs = ['classes_out', 'scores_out', 'boxes_out']
        if is_detection_out(self.launcher_config) and is_box_outputs(self.launcher_config, box_outputs):
            raise ConfigError('only detection output or [{}] should be provided'.format(', '.join(box_outputs)))

        self.raw_masks_out = self.get_value_from_config('raw_masks_out')

        if is_detection_out(self.launcher_config):
            self.detection_out = self.get_value_from_config('detection_out')
            self.realisation = self._process_detection_output
        else:
            self.classes_out = self.get_value_from_config('classes_out')
            self.scores_out = self.get_value_from_config('scores_out')
            self.boxes_out = self.get_value_from_config('boxes_out')
            self.num_detections_out = self.get_value_from_config('num_detections_out')

            if self.num_detections_out:
                if not is_box_outputs(self.launcher_config, box_outputs):
                    raise ConfigError('all related outputs should be specified: {}'.format(', '.join(box_outputs)))
                self.realisation = self._process_tf_obj_detection_api_outputs
                return

            self.realisation = self._process_pytorch_outputs

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        return self.realisation(raw_outputs, identifiers, frame_meta)

    def _process_tf_obj_detection_api_outputs(self, raw_outputs, identifiers, frame_meta):
        num_detections = raw_outputs[self.num_detections_out]
        classes = raw_outputs[self.classes_out]
        boxes = raw_outputs[self.boxes_out]
        scores = raw_outputs[self.scores_out]
        raw_masks = raw_outputs[self.raw_masks_out]

        results = []

        for identifier, image_meta, im_num_detections, im_classes, im_boxes, im_scores, im_raw_masks in zip(
                identifiers, frame_meta, num_detections, classes, boxes, scores, raw_masks
        ):
            num_valid_detections = int(im_num_detections)
            im_classes = im_classes[:num_valid_detections]
            im_scores = im_scores[:num_valid_detections]
            im_boxes = im_boxes[:num_valid_detections]
            im_raw_masks = im_raw_masks[:num_valid_detections]
            original_image_size = image_meta['image_size'][:2]
            im_boxes[:, 1::2] *= original_image_size[1]
            im_boxes[:, 0::2] *= original_image_size[0]
            im_classes = im_classes.astype(np.uint32)
            masks = []
            for box, raw_cls_mask in zip(im_boxes, im_raw_masks):
                box = np.array([box[1], box[0], box[3], box[2]])
                mask = self.segm_postprocess(box, raw_cls_mask, *original_image_size, True, True)
                masks.append(mask)
            y_mins, x_mins, y_maxs, x_maxs = im_boxes.T
            detection_prediction = DetectionPrediction(
                identifier, im_classes, im_scores, x_mins, y_mins, x_maxs, y_maxs
            )
            instance_segmentation_prediction = CoCoInstanceSegmentationPrediction(
                identifier, masks, im_classes, im_scores
            )
            instance_segmentation_prediction.metadata['rects'] = np.c_[x_mins, y_mins, x_maxs, y_maxs]
            instance_segmentation_prediction.metadata['image_size'] = image_meta['image_size']
            results.append(ContainerPrediction({
                'detection_prediction': detection_prediction,
                'segmentation_prediction': instance_segmentation_prediction
            }))

            return results

    def _process_pytorch_outputs(self, raw_outputs, identifiers, frame_meta):
        if self.boxes_out not in raw_outputs:
            self.boxes_out = self._find_output(raw_outputs)
            warnings.warn(
                'Using auto-detected output {} with bounding boxes.'.format(self.boxes_out)
            )

        boxes = raw_outputs[self.boxes_out]
        scores = raw_outputs.get(self.scores_out, None)
        classes = raw_outputs.get(self.classes_out, None)
        raw_masks = raw_outputs.get(self.raw_masks_out, None)

        if scores is None and boxes.ndim == 2 and boxes.shape[1] == 5:
            scores = boxes[:, 4]
            boxes = boxes[:, :4]

        if classes is None:
            classes = np.ones(len(boxes), np.uint32)

        valid_detections_mask = classes > 0 if self.scores_out else np.sum(boxes, axis=1) > 0
        classes = classes[valid_detections_mask]
        boxes = boxes[valid_detections_mask]
        scores = scores[valid_detections_mask]
        if raw_masks is not None:
            raw_masks = raw_masks[valid_detections_mask]

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
                image_input = image_input[0]
                if image_input[1] == 3:
                    processed_image_size = image_input[2:]
                else:
                    processed_image_size = image_input[1:3]
                im_scale_y = processed_image_size[0] / original_image_size[0]
                im_scale_x = processed_image_size[1] / original_image_size[1]
            boxes[:, 0::2] /= im_scale_x
            boxes[:, 1::2] /= im_scale_y
            classes = classes.astype(np.uint32)
            masks = []
            if raw_masks is not None:
                masks = self._process_masks_pytorch(boxes, raw_masks, identifiers, original_image_size, classes)

            x_mins, y_mins, x_maxs, y_maxs = boxes.T
            detection_prediction = DetectionPrediction(identifier, classes, scores, x_mins, y_mins, x_maxs, y_maxs)
            instance_segmentation_prediction = CoCoInstanceSegmentationPrediction(identifier, masks, classes, scores)
            instance_segmentation_prediction.metadata['rects'] = np.c_[x_mins, y_mins, x_maxs, y_maxs]
            instance_segmentation_prediction.metadata['image_size'] = image_meta['image_size']
            results.append(ContainerPrediction({
                'detection_prediction': detection_prediction,
                'segmentation_prediction': instance_segmentation_prediction
            }))

            return results

    @staticmethod
    def _find_output(predictions):
        filter_outputs = [
            output_name for output_name, out_data in predictions.items()
            if len(np.shape(out_data)) == 2 and np.shape(out_data)[-1] == 5
        ]
        if not filter_outputs:
            raise ConfigError('Suitable output layer not found')
        if len(filter_outputs) > 1:
            warnings.warn(
                'There are several suitable outputs {}. The first will be used. '.format(', '.join(filter_outputs)) +
                'If you need to use another layer, please specify it explicitly.'
            )
        return filter_outputs[0]

    def _process_masks_pytorch(self, boxes, raw_masks, identifiers, original_image_size, classes):
        masks = []
        raw_mask_for_all_classes = np.shape(raw_masks)[1] != len(identifiers)
        if raw_mask_for_all_classes:
            per_obj_raw_masks = []
            for cls, raw_mask in zip(classes, raw_masks):
                per_obj_raw_masks.append(raw_mask[cls, ...] if self.scores_out else raw_mask)
        else:
            per_obj_raw_masks = np.squeeze(raw_masks, axis=1)

        for box, raw_cls_mask in zip(boxes, per_obj_raw_masks):
            mask = self.segm_postprocess(box, raw_cls_mask, *original_image_size, True, True)
            masks.append(mask)
        return masks

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
            coeff_x, coeff_y = FRCNNPostprocessingBboxResize.get_coeff_x_y_from_metadata(frame_meta[batch_index])
            prediction_box_mask = np.where(detections_boxes[:, 0] == batch_index)
            filtered_detections_boxes = detections_boxes[prediction_box_mask]
            filtered_detections_boxes = filtered_detections_boxes[:, 1::]
            filtered_masks = raw_masks[prediction_box_mask]
            detection_prediction = DetectionPrediction(identifier, *zip(*filtered_detections_boxes))
            instance_masks = []
            for box, masks in zip(filtered_detections_boxes, filtered_masks):
                label = box[0]
                cls_mask = masks[int(label) - 1, ...]
                box[2::2] *= coeff_x
                box[3::2] *= coeff_y
                cls_mask = self.segm_postprocess(box[2:], cls_mask, *image_size, True, True)
                instance_masks.append(cls_mask)
            instance_segmentation_prediction = CoCoInstanceSegmentationPrediction(
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
        w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)  # pylint: disable=E0633
        x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
        x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

        raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
        mask = raw_cls_mask.astype(np.uint8)

        if full_image_mask:
            # Put an object mask in an image mask.
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
            mask_start_y = y0 - extended_box[1]
            mask_end_y = y1 - extended_box[1]
            mask_start_x = x0 - extended_box[0]
            mask_end_x = x1 - extended_box[0]
            im_mask[y0:y1, x0:x1] = mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x]
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
