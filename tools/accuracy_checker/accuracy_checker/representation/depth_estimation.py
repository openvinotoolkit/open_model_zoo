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

from enum import Enum
import numpy as np

from .base_representation import BaseRepresentation
from ..data_readers import BaseReader


class GTLoader(Enum):
    PILLOW = 0
    OPENCV = 1
    NUMPY = 2


class DepthEstimationRepresentation(BaseRepresentation):
    pass


class DepthEstimationAnnotation(DepthEstimationRepresentation):
    LOADERS = {
        GTLoader.PILLOW: 'pillow_imread',
        GTLoader.OPENCV: {'type': 'opencv_imread', 'reading_flag': 'unchanged'},
        GTLoader.NUMPY: 'numpy_reader'
    }

    def __init__(self, identifier, depth_map_path, gt_loader=GTLoader.OPENCV):
        super().__init__(identifier)
        self._depth_map_path = depth_map_path
        self._gt_loader = gt_loader
        self._depth_map = None
        self._mask = None

    def _load_depth_map(self):
        data_source = self.metadata.get('additional_data_source')
        if data_source is None:
            data_source = self.metadata['data_source']
        loader_config = self.LOADERS.get(self._gt_loader)
        if isinstance(loader_config, str):
            loader = BaseReader.provide(loader_config, data_source)
        else:
            loader = BaseReader.provide(loader_config['type'], data_source, config=loader_config)
        if self._gt_loader == GTLoader.PILLOW:
            loader.convert_to_rgb = False
        input_map = loader.read(self._depth_map_path)
        if self._gt_loader != GTLoader.NUMPY:
            self._depth_map = 1 - input_map / 255
            self._mask = input_map != 255
        else:
            self._depth_map = input_map
            self._mask = np.ones_like(input_map, dtype=bool)

    @property
    def depth_map(self):
        if self._depth_map is None:
            self._load_depth_map()
        return self._depth_map

    @property
    def mask(self):
        if self._mask is None:
            self._load_depth_map()
        return self._mask


class DepthEstimationPrediction(DepthEstimationRepresentation):
    def __init__(self, identifier, depth_map):
        super().__init__(identifier)
        self.depth_map = depth_map
