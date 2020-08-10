"""
Copyright (c) 2018-2020 Intel Corporation

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

from ..data_readers import BaseReader
from .image_processing import ImageProcessingAnnotation, ImageProcessingPrediction


class SuperResolutionAnnotation(ImageProcessingAnnotation):
    @property
    def value(self):
        if self._value is None:
            data_source = self.metadata.get('additional_data_source')
            if not data_source:
                data_source = self.metadata['data_source']
            loader = BaseReader.provide(self._gt_loader, data_source)
            gt = loader.read(self._image_path)
            return gt.astype(np.uint8) if self._gt_loader != 'dicom_reader' else gt
        return self._value


class SuperResolutionPrediction(ImageProcessingPrediction):
    pass
