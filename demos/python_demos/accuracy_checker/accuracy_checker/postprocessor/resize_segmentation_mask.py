"""
Copyright (c) 2018 Intel Corporation

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

import scipy.misc
import numpy as np
from ..config import NumberField
from ..utils import  get_size_from_config
from .postprocessor import PostprocessorWithSpecificTargets, PostprocessorWithTargetsConfigValidator
from ..representation import SegmentationPrediction, SegmentationAnnotation


class ResizeSegmentationMask(PostprocessorWithSpecificTargets):
    __provider__ = 'resize_segmentation_mask'
    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )

    def validate_config(self):
        class _ResizeConfigValidator(PostprocessorWithTargetsConfigValidator):
            size = NumberField(floats=False, optional=True)
            dst_width = NumberField(floats=False, optional=True)
            dst_height = NumberField(floats=False, optional=True)

        resize_config_validator = _ResizeConfigValidator(self.__provider__)
        resize_config_validator.validate(self.config)

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)

    def process_image(self, annotation, prediction):
        target_width = self.image_size[1] if self.dst_width is None else self.dst_width
        target_height = self.image_size[0] if self.dst_height is None else self.dst_height

        for target in annotation:
            self._resize_annotation_segmentation_mask(target, target_height, target_width)

        for target in prediction:
            self._resize_prediction_segmentation_mask(target, target_height, target_width)

        return annotation, prediction

    @staticmethod
    def _resize_prediction_segmentation_mask(entry, height, width):
        entry_mask = []
        for class_mask in entry.mask:
            resized_mask = scipy.misc.imresize(class_mask, (height, width), 'nearest')
            entry_mask.append(resized_mask)
        entry.mask = np.array(entry_mask)

        return entry

    @staticmethod
    def _resize_annotation_segmentation_mask(entry, height, width):
        entry.mask = scipy.misc.imresize(entry.mask, (height, width), 'nearest')
        return entry
