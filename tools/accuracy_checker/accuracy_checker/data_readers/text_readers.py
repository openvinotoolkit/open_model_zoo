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

import numpy as np

from ..config import StringField, ConfigError
from .data_reader import BaseReader
from ..utils import get_path, read_json


class JSONReader(BaseReader):
    __provider__ = 'json_reader'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'key': StringField(optional=True, case_sensitive=True,
                               description='Key for reading from json dictionary.')
        })
        return parameters

    def configure(self):
        self.key = self.get_value_from_config('key')
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.data_layout = self.get_value_from_config('data_layout')
        if not self.data_source:
            if not self._postpone_data_source:
                raise ConfigError('data_source parameter is required to create "{}" '
                                  'data reader and read data'.format(self.__provider__))
        else:
            self.data_source = get_path(self.data_source, is_directory=True)

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        data = read_json(data_path)
        if self.key:
            data = data.get(self.key)

            if not data:
                raise ConfigError('{} does not contain {}'.format(data_id, self.key))

        return np.array(data).astype(np.float32)
