"""
Copyright (c) 2018-2022 Intel Corporation

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
from .postprocessor import Postprocessor


class SalientMapNormalizer(Postprocessor):
    __provider__ = 'normalize_salience_map'

    def process_image(self, annotation, prediction):
        for ann in annotation:
            gt_mask = ann.mask
            if len(gt_mask.shape) == 3 and gt_mask.shape[-1] == 3:
                gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            gt_mask = gt_mask / 255
            gt_mask[gt_mask >= 0.5] = 1
            gt_mask[gt_mask < 0.5] = 0
            ann.mask = gt_mask.astype(np.uint8)
        return annotation, prediction
