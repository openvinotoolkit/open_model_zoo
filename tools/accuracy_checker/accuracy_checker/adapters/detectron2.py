"""
Copyright (c) 2026 Intel Corporation

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
from ..representation import CoCoInstanceSegmentationPrediction, DetectionPrediction, ContainerPrediction


class Detectron2Adapter(MaskRCNNAdapter):
    __provider__ = 'detectron2'

    @staticmethod
    def _is_roi_masks(pred_masks):
        arr = np.asarray(pred_masks)
        return arr.ndim == 4 and arr.shape[1] == 1 and arr.shape[2] <= 64 and arr.shape[3] <= 64

    def _auto_resolve_outputs(self, raw_outputs):
        # Keep user-provided names if they exist in current outputs.
        boxes_name = self.boxes_out if self.boxes_out in raw_outputs else None
        classes_name = self.classes_out if self.classes_out in raw_outputs else None
        scores_name = self.scores_out if self.scores_out in raw_outputs else None
        masks_name = self.raw_masks_out if self.raw_masks_out in raw_outputs else None

        items = list(raw_outputs.items())

        def _remove_selected(name):
            if name is None:
                return
            for idx, (candidate_name, _) in enumerate(items):
                if candidate_name == name:
                    items.pop(idx)
                    return

        def _pick_and_remove(predicate):
            for idx, (name, value) in enumerate(items):
                if predicate(name, value):
                    items.pop(idx)
                    return name
            return None

        # Exclude explicitly selected outputs from further auto-matching.
        _remove_selected(boxes_name)
        _remove_selected(classes_name)
        _remove_selected(scores_name)
        _remove_selected(masks_name)

        # Detect boxes: [N, 4]
        if boxes_name is None:
            boxes_name = _pick_and_remove(
                lambda _, value: np.asarray(value).ndim == 2 and np.asarray(value).shape[1] == 4
            )

        # Detect masks: [N, 1, H, W] or [N, H, W]
        if masks_name is None:
            masks_name = _pick_and_remove(
                lambda _, value: np.asarray(value).ndim == 4 and np.asarray(value).shape[1] == 1
            )
            if masks_name is None:
                masks_name = _pick_and_remove(
                    lambda _, value: np.asarray(value).ndim == 3
                )

        # Detect classes: integer [N]
        if classes_name is None:
            classes_name = _pick_and_remove(
                lambda _, value: np.asarray(value).ndim == 1
                and np.issubdtype(np.asarray(value).dtype, np.integer)
                and np.asarray(value).size != 2
            )

        # Detect scores: float [N]
        if scores_name is None:
            scores_name = _pick_and_remove(
                lambda _, value: np.asarray(value).ndim == 1 and np.issubdtype(np.asarray(value).dtype, np.floating)
            )

        if boxes_name is None:
            raise ConfigError('Suitable output layer not found')

        self.boxes_out = boxes_name
        self.classes_out = classes_name
        self.scores_out = scores_name
        self.raw_masks_out = masks_name

    def _process_pytorch_outputs(self, raw_outputs, identifiers, frame_meta):
        self._auto_resolve_outputs(raw_outputs)

        boxes = np.asarray(raw_outputs[self.boxes_out])
        scores = raw_outputs.get(self.scores_out, None)
        classes = raw_outputs.get(self.classes_out, None)
        pred_masks = raw_outputs.get(self.raw_masks_out, None)

        scores = np.asarray(scores) if scores is not None else None
        classes = np.asarray(classes) if classes is not None else None
        pred_masks = np.asarray(pred_masks) if pred_masks is not None else None

        if scores is None and boxes.ndim == 2 and boxes.shape[1] == 5:
            scores = boxes[:, 4]
            boxes = boxes[:, :4]

        if classes is None:
            classes = np.ones(len(boxes), np.uint32)

        if scores is not None:
            valid_detections_mask = scores > 0
        else:
            valid_detections_mask = np.sum(boxes, axis=1) > 0
        classes = classes[valid_detections_mask]
        boxes = boxes[valid_detections_mask]
        scores = scores[valid_detections_mask]
        if pred_masks is not None:
            pred_masks = pred_masks[valid_detections_mask]

        results = []

        for identifier, image_meta in zip(identifiers, frame_meta):
            original_image_size = image_meta['image_size'][:2]
            
            # Rescale boxes to original image space using parent's logic
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
            
            classes = classes.astype(np.int32)
            # Some exported pipelines emit 1-based classes; COCO 80cl in this setup is 0-based.
            if classes.size and np.min(classes) >= 1:
                classes = classes - 1
            classes = classes.astype(np.uint32)

            masks = []
            if pred_masks is not None:
                if self._is_roi_masks(pred_masks):
                    masks = self._process_masks_pytorch(boxes, pred_masks, identifiers, original_image_size, classes)
                else:
                    masks = self._process_detectron2_masks(pred_masks, original_image_size)

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

    def _process_detectron2_masks(self, pred_masks, original_image_size):
        masks = []
        original_height, original_width = original_image_size

        for mask in pred_masks:
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim == 3:
                mask = np.squeeze(mask, axis=0)

            if mask.shape != (original_height, original_width):
                mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

            mask = (mask > 0.5).astype(np.uint8)
            encoded_mask = self.encoder(np.array(mask[:, :, np.newaxis], order='F'))[0]
            encoded_mask['counts'] = encoded_mask['counts'].decode('utf-8')
            masks.append(encoded_mask)

        return masks