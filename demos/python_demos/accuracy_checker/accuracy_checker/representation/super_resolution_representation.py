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
import cv2
import numpy as np

from .base_representation import BaseRepresentation


class GTLoader(Enum):
    PIL = 0
    OPENCV = 1

LOADER_FUNCTORS = {
    GTLoader.PIL: Image.open,
    GTLoader.OPENCV: cv2.imread
}


class SuperResolutionRepresentation(BaseRepresentation):
    pass


class SuperResolutionAnnotation(SuperResolutionRepresentation):
    def __init__(self, identifier, path_to_hr, gt_loader=GTLoader.PIL):
        """
        Args:
            identifier: object identifier (e.g. image name)
            path_to_hr: path where hight resolution image should be loaded from. The path is relative to data source
            gt_loader: back-end, used to load segmentation masks
        """
        super().__init__(identifier)
        self._image_path = path_to_hr
        gt_loader_func = LOADER_FUNCTORS.get(gt_loader)
        if not gt_loader_func:
            raise RuntimeError("Unknown ground truth loader type {}".format(gt_loader))
        self._gt_loader = gt_loader_func

    @property
    def value(self):
        return np.array(self._gt_loader(str(self.metadata['data_source'] / self._image_path)), dtype=np.uint8)


class SuperResolutionPrediction(SuperResolutionRepresentation):
    def __init__(self, identifiers, prediction):
        """
        Args:
            identifiers: object identifier (e.g. image name)
            prediction: array with shape (height, width) contained result image
        """
        super().__init__(identifiers)
        self.value = prediction
