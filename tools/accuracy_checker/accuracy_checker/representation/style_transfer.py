"""
Copyright (c) 2020 Intel Corporation

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
import numpy as np

from .base_representation import BaseRepresentation
from ..data_readers import BaseReader

class GTLoader(Enum):
    PILLOW = 0
    OPENCV = 1


class StyleTransferRepresentation(BaseRepresentation):
    pass


class StyleTransferAnnotation(StyleTransferRepresentation):
    LOADERS = {
        GTLoader.PILLOW: 'pillow_imread',
        GTLoader.OPENCV: 'opencv_imread'
    }

    def __init__(self, identifier, image_path, gt_loader=GTLoader.PILLOW):
        super().__init__(identifier)
        self._gt_loader = self.LOADERS.get(gt_loader)
        self._image_path = image_path
        self._value = None

    @property
    def value(self):
        if self._value is not None:
            return self._value

        loader = BaseReader.provide(self._gt_loader, self.metadata['data_source'])
        gt = loader.read(self._image_path)
        return gt.astype(np.uint8)

    @value.setter
    def value(self, val):
        self._value = val
        return self._value

class StyleTransferPrediction(StyleTransferRepresentation):
    def __init__(self, identifiers, prediction):
        super().__init__(identifiers)
        self.value = prediction
