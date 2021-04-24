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

import numpy as np

from .postprocessor import PostprocessorWithSpecificTargets
from ..representation import SegmentationAnnotation, SegmentationPrediction


class EncodeSegMask(PostprocessorWithSpecificTargets):
    """
    Encode segmentation label image as segmentation mask.
    """

    __provider__ = 'encode_segmentation_mask'

    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )

    def process_image(self, annotation, prediction):
        segmentation_colors = self.meta.get("segmentation_colors")
        prediction_to_gt_label = self.meta.get('prediction_to_gt_labels')

        if annotation and any([ann is not None for ann in annotation]):
            if not segmentation_colors:
                raise ValueError("No 'segmentation_colors' in dataset metadata.")

        if prediction:
            if not prediction_to_gt_label and not self._deprocess_predictions:
                raise ValueError("No 'prediction_to_gt_labels' in dataset metadata")

        for annotation_ in annotation:
            if annotation_ is None:
                continue
            mask = annotation_.mask.astype(int)
            num_channels = len(mask.shape)
            encoded_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
            for label, color in enumerate(segmentation_colors):
                encoded_mask[np.where(
                    np.all(mask == color, axis=-1) if num_channels >= 3 else mask == color
                )[:2]] = label
            annotation_.mask = encoded_mask.astype(np.int8)

        if not self._deprocess_predictions:
            for prediction_ in prediction:
                if prediction_ is None:
                    continue
                mask = prediction_.mask
                updated_mask = mask
                saved_mask = mask.copy()
                if len(mask.shape) == 3 and mask.shape[0] != 1:
                    updated_mask = np.argmax(mask, axis=0)
                    saved_mask = updated_mask.copy()
                for pred_label, gt_label in prediction_to_gt_label.items():
                    updated_mask[saved_mask == pred_label] = gt_label

                updated_mask[saved_mask >= len(prediction_to_gt_label)] = 255

                prediction_.mask = updated_mask.astype(np.int8)
        self._deprocess_predictions = False

        return annotation, prediction
