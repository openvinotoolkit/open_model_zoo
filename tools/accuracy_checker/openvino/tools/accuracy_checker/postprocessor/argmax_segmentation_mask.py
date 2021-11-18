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
from ..representation import SegmentationAnnotation, SegmentationPrediction


class ArgMaxSegmentationMask(Postprocessor):
    __provider__ = 'argmax_segmentation_mask'
    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )

    def process_image(self, annotation, prediction):
        for ann in annotation:
            ann.mask = np.argmax(ann.mask, axis=-1)
        return annotation, prediction
