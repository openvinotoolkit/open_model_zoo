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

import numpy as np
from .postprocessor import PostprocessorWithSpecificTargets
from ..representation import BrainTumorSegmentationAnnotation, BrainTumorSegmentationPrediction
from ..config import NumberField, ConfigError


class ClipSegmentationMask(PostprocessorWithSpecificTargets):
    __provider__ = 'clip_segmentation_mask'

    annotation_types = (BrainTumorSegmentationAnnotation, )
    prediction_types = (BrainTumorSegmentationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'min_value': NumberField(value_type=int, min_value=0, optional=True, default=0, description="Min value"),
            'max_value': NumberField(value_type=int, description="Max value")
        })
        return parameters

    def configure(self):
        self.min_value = self.get_value_from_config('min_value')
        self.max_value = self.get_value_from_config('max_value')
        if self.max_value < self.min_value:
            raise ConfigError('max_value should be greater than min_value')

    def process_image(self, annotation, prediction):
        for target in annotation:
            target.mask = np.clip(target.mask, a_min=self.min_value, a_max=self.max_value)

        for target in prediction:
            target.mask = np.clip(target.mask, a_min=self.min_value, a_max=self.max_value)

        return annotation, prediction
