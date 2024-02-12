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

import cv2
import numpy as np

from .data_reader import BaseReader, DataRepresentation
from ..utils import read_pickle, UnsupportedPackage

try:
    import lmdb
except ImportError as import_error:
    lmdb = UnsupportedPackage("lmdb", import_error.msg)


class PickleReader(BaseReader):
    __provider__ = 'pickle_reader'

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        data = read_pickle(data_path)
        if isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict):
            return data
        return data, {}

    def read_item(self, data_id):
        data = DataRepresentation(*self.read_dispatcher(data_id), identifier=data_id)
        if self.multi_infer:
            data.metadata['multi_infer'] = self.multi_infer
        if self.data_layout:
            data.metadata['data_layout'] = self.data_layout
        return data


class ByteFileReader(BaseReader):
    __provider__ = 'byte_reader'

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        with open(data_path, 'rb') as f:
            return np.array(f.read())


class LMDBReader(BaseReader):
    __provider__ = 'lmdb_reader'

    def configure(self):
        super().configure()
        if isinstance(lmdb, UnsupportedPackage):
            lmdb.raise_error(self.__provider__)
        self.database = lmdb.open(bytes(self.data_source), readonly=True, lock=False)

    def read(self, data_id):
        with self.database.begin(write=False) as txn:
            img_key = f'image-{data_id:09d}'.encode()
            image_bytes = txn.get(img_key)
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if len(img.shape) < 3:
                img = np.stack((img,) * 3, axis=-1)
            assert img.shape[-1] == 3
            return img
