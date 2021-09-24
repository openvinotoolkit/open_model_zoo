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
import cv2
from .postprocessor import Postprocessor
from ..representation import (
    SuperResolutionPrediction, SuperResolutionAnnotation,
    ImageProcessingAnnotation, ImageProcessingPrediction,
    StyleTransferAnnotation, StyleTransferPrediction
)


class RGB2GRAYAnnotation(Postprocessor):
    __provider__ = 'rgb_to_gray'

    annotation_types = (SuperResolutionAnnotation, ImageProcessingAnnotation, StyleTransferAnnotation)
    prediction_types = (SuperResolutionPrediction, ImageProcessingPrediction, StyleTransferPrediction)

    def process_image(self, annotation, prediction):
        for annotation_ in annotation:
            annotation_.value = np.expand_dims(cv2.cvtColor(annotation_.value, cv2.COLOR_RGB2GRAY), -1)

        return annotation, prediction


class BGR2GRAYAnnotation(Postprocessor):
    __provider__ = 'bgr_to_gray'

    annotation_types = (SuperResolutionAnnotation, ImageProcessingAnnotation, StyleTransferAnnotation)
    prediction_types = (SuperResolutionPrediction, ImageProcessingPrediction, StyleTransferPrediction)

    def process_image(self, annotation, prediction):
        for annotation_ in annotation:
            annotation_.value = np.expand_dims(cv2.cvtColor(annotation_.value, cv2.COLOR_BGR2GRAY), -1)

        return annotation, prediction
