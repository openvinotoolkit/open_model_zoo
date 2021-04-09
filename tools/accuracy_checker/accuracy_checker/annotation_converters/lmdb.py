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
import cv2

from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn
from ..utils import UnsupportedPackage
from ..representation import CharacterRecognitionAnnotation
from ..config import BoolField

try:
    import lmdb
except ImportError as import_error:
    lmdb = UnsupportedPackage("lmdb", import_error.msg)


class LMDBConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'lmdb_text_recognition_database'
    annotation_types = (CharacterRecognitionAnnotation, )
    supported_symbols = '0123456789abcdefghijklmnopqrstuvwxyz'

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'lower_case': BoolField(description='Convert GT text to lowercase.', optional=True)
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.lower_case = self.get_value_from_config('lower_case')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        """Reads data from disk and returns dataset in converted for AC format

        Args:
            check_content (bool, optional): Check if content is valid. Defaults to False.
            progress_callback (bool, optional): Display progress. Defaults to None.
            progress_interval (int, optional): Units to display progress. Defaults to 100 (percent).

        Returns:
            [type]: Converted dataset
        """
        annotations = []
        content_errors = None if not check_content else []
        lmdb_env = lmdb.open(bytes(self.data_dir), readonly=True)
        with lmdb_env.begin(write=False) as txn:
            num_iterations = int(txn.get('num-samples'.encode()))
            for index in range(1, num_iterations + 1):
                label_key = f'label-{index:09d}'.encode()
                text = txn.get(label_key).decode('utf-8')
                if self.lower_case:
                    text = text.lower()
                if progress_callback is not None and index % progress_interval == 0:
                    progress_callback(index / num_iterations * 100)
                if check_content:
                    img_key = f'label-{index:09d}'.encode()
                    image_bytes = txn.get(img_key)
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_ANYCOLOR)
                    if image is None:
                        content_errors.append(f'label-{index:09d}: does not exist')
                annotations.append(CharacterRecognitionAnnotation(index, text))

        label_map = {ind: str(key) for ind, key in enumerate(self.supported_symbols)}
        meta = {'label_map': label_map, 'blank_label': len(label_map)}
        return ConverterReturn(annotations, meta, content_errors)
