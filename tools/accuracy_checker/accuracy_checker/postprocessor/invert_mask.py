"""
Copyright (c) 2018-2024 Intel Corporation

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
from .postprocessor import PostprocessorWithSpecificTargets
from  ..representation import (
    SegmentationPrediction, SegmentationAnnotation, BackgroundMattingAnnotation, BackgroundMattingPrediction
)


class InvertMask(PostprocessorWithSpecificTargets):
    __provider__ = 'invert_mask'
    annotation_types = (SegmentationAnnotation, BackgroundMattingAnnotation)
    prediction_types = (SegmentationPrediction, BackgroundMattingPrediction)

    def process_image(self, annotation, prediction):
        for ann in annotation:
            ann.mask = cv2.bitwise_not(ann.mask)

        for pred in prediction:
            pred.mask = cv2.bitwise_not(pred.mask)

        return annotation, prediction
