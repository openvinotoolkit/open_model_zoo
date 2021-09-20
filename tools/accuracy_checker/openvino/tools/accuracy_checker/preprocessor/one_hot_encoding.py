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
from ..config import NumberField
from .preprocessor import Preprocessor


class OneHotEncoding(Preprocessor):
    __provider__ = 'one_hot_encoding'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'value': NumberField(
                optional=True,
                value_type=int,
                default=1,
                description="Number for label encoding. Integer which used to indicate label"
            ),
            'number_of_classes': NumberField(
                optional=True,
                value_type=int,
                default=1,
                description="Number of encoding classes"
            ),
            'axis': NumberField(
                optional=True,
                value_type=int,
                default=1,
                description="Axis for encoding"
            ),
            'base': NumberField(
                optional=True,
                value_type=int,
                default=0,
                description="Number that specifies the values of the other classes at one hot label map"
            )
        })
        return parameters

    def configure(self):
        self.value = self.get_value_from_config('value')
        self.base = self.get_value_from_config('base')
        self.classes = self.get_value_from_config('number_of_classes')
        self.axis = self.get_value_from_config('axis')

    def process(self, image, annotation_meta=None):
        def process_data(data, classes, axis, value, base):
            data = data.astype(np.int32)
            shapes = list(data.shape)
            ndim = len(shapes)
            shapes[axis] = classes
            base_arr = np.full(shapes, base, np.int)
            expanded_index = []
            for i in range(ndim):
                arr = (data if axis == i
                       else np.arange(shapes[i]).reshape([shapes[i] if i == j else 1 for j in range(ndim)]))
                expanded_index.append(arr)
            base_arr[tuple(expanded_index)] = value
            return base_arr

        image.data = (process_data(image.data, self.classes, self.axis, self.value, self.base)
                      if not isinstance(image.data, list) else [
                          process_data(data_fragment, self.classes, self.axis, self.value, self.base)
                          for data_fragment in image.data])

        return image
