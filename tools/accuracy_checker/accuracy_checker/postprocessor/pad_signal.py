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
from .postprocessor import Postprocessor
from ..representation import NoiseSuppressionAnnotation, NoiseSuppressionPrediction


class PadSignal(Postprocessor):
    __provider__ = 'pad_signal'
    annotation_types = (NoiseSuppressionAnnotation, )
    prediction_types = (NoiseSuppressionPrediction, )

    def process_image_with_metadata(self, annotation, prediction, image_metadata=None):
        for ann in annotation:
            padding = image_metadata.get('padding')
            if padding is None:
                continue
            ann.value = np.pad(ann.value, (padding, 0), mode='constant')
        return annotation, prediction

    def process_image(self, annotation, prediction):
        return annotation, prediction
