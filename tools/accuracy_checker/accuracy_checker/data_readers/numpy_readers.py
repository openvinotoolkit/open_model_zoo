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

import re
from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile

from ..config import StringField, BoolField, NumberField, ConfigError
from .data_reader import BaseReader, DataRepresentation
from ..utils import get_path


class NumPyReader(BaseReader):
    __provider__ = 'numpy_reader'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'keys': StringField(optional=True, default="", description='Comma-separated model input names.'),
            'separator': StringField(optional=True,
                                     description='Separator symbol between input identifier and file identifier.'),
            'id_sep': StringField(
                optional=True, default="_",
                description='Separator symbol between input name and record number in input identifier.'
            ),
            'block': BoolField(optional=True, default=False, description='Allows block mode.'),
            'batch': NumberField(optional=True, default=1, description='Batch size'),
            'records_mode': BoolField(optional=True, default=False, description='separate data on records'),
        })
        return parameters

    def configure(self):
        self.is_text = self.config.get('text_file', False)
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.keys = self.get_value_from_config('keys')
        self.keys = [t.strip() for t in self.keys.split(',')] if len(self.keys) > 0 else []
        self.separator = self.get_value_from_config('separator')
        self.id_sep = self.get_value_from_config('id_sep')
        self.block = self.get_value_from_config('block')
        self.batch = int(self.get_value_from_config('batch'))
        self.record_mode = self.get_value_from_config('records_mode')

        if self.separator and self.is_text:
            raise ConfigError('text file reading with numpy does support separation')
        if not self.data_source:
            if not self._postpone_data_source:
                raise ConfigError('data_source parameter is required to create "{}" '
                                  'data reader and read data'.format(self.__provider__))
        else:
            self.data_source = get_path(self.data_source, is_directory=True)
        self.keyRegex = {k: re.compile(k + self.id_sep) for k in self.keys}
        self.valRegex = re.compile(r"([^0-9]+)([0-9]+)")
        self.data_layout = self.get_value_from_config('data_layout')

    def read(self, data_id):
        field_id = None
        if self.separator:
            field_id, data_id = str(data_id).split(self.separator)
        data_path = self.data_source / data_id if self.data_source is not None else data_id

        data = np.load(str(data_path))

        if not isinstance(data, NpzFile):
            return data

        if field_id is not None:
            key = [k for k, v in self.keyRegex.items() if v.match(field_id)]
            if len(key) > 0:
                if self.block:
                    res = data[key[0]]
                else:
                    recno = field_id.split('_')[-1]
                    recno = int(recno)
                    start = Path(data_id).name.split('.')[0]
                    start = int(start)
                    res = data[key[0]][recno - start, :]
                return res

        key = next(iter(data.keys()))
        data = data[key]
        if self.record_mode and self.id_sep in field_id:
            recno = field_id.split(self.id_sep)[-1]
            recno = int(recno)
            res = data[recno, :]
            return res
        if self.multi_infer:
            return list(data)
        return data


class NumpyTXTReader(BaseReader):
    __provider__ = 'numpy_txt_reader'

    def read(self, data_id):
        return np.loadtxt(str(self.data_source / data_id))


class NumpyDictReader(BaseReader):
    __provider__ = 'numpy_dict_reader'

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        return np.load(str(data_path), allow_pickle=True)[()]

    def read_item(self, data_id):
        dict_data = self.read_dispatcher(data_id)
        identifier = []
        data = []
        for key, value in dict_data.items():
            identifier.append('{}.{}'.format(data_id, key))
            data.append(value)
        if len(data) == 1:
            return DataRepresentation(data[0], identifier=data_id)
        return DataRepresentation(data, identifier=identifier)


class NumpyBinReader(BaseReader):
    __provider__ = 'numpy_bin_reader'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            "dtype": StringField(optional=True, default='float32', description='data type for reading'),
            'as_buffer': BoolField(optional=True, default=False, description='interpter binary data as buffere'),
            'offset': NumberField(optional=True, default=0, value_type=int, min_value=0)
        })
        return params

    def configure(self):
        super().configure()
        self.dtype = self.get_value_from_config('dtype')
        self.as_buffer = self.get_value_from_config('as_buffer')
        self.offset = self.get_value_from_config('offset')
        self.data_layout = self.get_value_from_config('data_layout')

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        if not self.as_buffer:
            return np.fromfile(data_path, dtype=self.dtype)
        buffer = Path(data_path).open('rb').read()
        return np.frombuffer(buffer[self.offset:], dtype=self.dtype)
