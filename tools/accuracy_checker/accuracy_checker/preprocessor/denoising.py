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

import numpy as np
try:
    import scipy
except ImportError:
    scipy = None

from .preprocessor import Preprocessor
from ..config import ConfigError


class GenerateNoiseMap(Preprocessor):
    __provider__ = 'generate_noise_map'

    def configure(self):
        if scipy is None:
            ConfigError(
                'Preprocessor *{}* require scipy installation. '.format(self.__provider__) +
                'Please install it before usage.'
            )

    def process(self, image, annotation_meta=None):
        img_smooth = scipy.ndimage.gaussian_filter(image.data, sigma=5)
        noise_map = -1.0 / np.power(10, 7) * img_smooth * img_smooth + 0.0034 * img_smooth + 3.6899
        image.data = np.stack([image.data, noise_map], axis=2)
        return image
