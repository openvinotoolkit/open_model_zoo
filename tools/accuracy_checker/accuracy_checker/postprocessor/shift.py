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
from ..config import NumberField
from .postprocessor import PostprocessorWithSpecificTargets
from ..representation import SegmentationAnnotation, SegmentationPrediction


class Shift(PostprocessorWithSpecificTargets):
    """
    Shift prediction segmentation mask or annotation segmentation mask.
    """

    __provider__ = 'shift'

    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )
    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'shift_value': NumberField(
                optional=True, value_type=int, default=0, description="Value for shift."
            )
        })
        return parameters

    def configure(self):
        self.shift_value = self.config.get('shift_value')

    def process_image(self, annotation, prediction):

        for annotation_ in annotation:
            mask = annotation_.mask.astype(int)
            update_mask = mask + self.shift_value
            annotation_.mask = update_mask.astype(np.int16)

        for prediction_ in prediction:
            mask = prediction_.mask
            update_mask = mask + self.shift_value
            prediction_.mask = update_mask.astype(np.int16)

        return annotation, prediction
