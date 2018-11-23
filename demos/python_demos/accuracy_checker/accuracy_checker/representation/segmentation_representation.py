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
from enum import Enum

from PIL import Image
import scipy.misc
import cv2
import numpy as np

from .base_representation import BaseRepresentation


class GTMaskLoader(Enum):
    PIL = 0
    SCIPY = 1
    OPENCV = 2


class SegmentationRepresentation(BaseRepresentation):
    pass


class SegmentationAnnotation(SegmentationRepresentation):
    def __init__(self, identifier, path_to_mask, mask_loader=GTMaskLoader.PIL):
        """
        Args:
            identifier: object identifier (e.g. image name)
            path_to_mask: path where segmentation mask should be loaded from. The path is relative to data source
            mask_loader: back-end, used to load segmentation masks
        """
        super().__init__(identifier)
        self._mask_path = path_to_mask
        self._mask_loader = mask_loader
        self._mask = None

    @property
    def mask(self):
        if self._mask is None:
            return self._load_mask()
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    def _load_mask(self):
        if self._mask_loader == GTMaskLoader.PIL:
            mask = Image.open(self._mask_path)
        elif self._mask_loader == GTMaskLoader.SCIPY:
            mask = scipy.misc.imread(self._mask_path)
        elif self._mask_loader == GTMaskLoader.OPENCV:
            mask = cv2.imread(self._mask_path)
        else:
            raise RuntimeError("Unknown Mask Loader type")
        return np.array(mask, dtype=np.uint8)


class SegmentationPrediction(SegmentationRepresentation):
    def __init__(self, identifiers, mask):
        """
        Args:
            identifiers: object identifier (e.g. image name)
            mask: array with shape (n_classes, height, width) of probabilities at each location
        """
        super().__init__(identifiers)
        self.mask = mask
