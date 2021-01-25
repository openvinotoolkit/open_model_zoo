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

from typing import Union
from pathlib import Path
from ..representation import CharacterRecognitionAnnotation
from ..utils import read_txt, check_file_existence, get_path
from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..config import PathField


class UnicodeCharacterRecognitionDatasetConverter(FileBasedAnnotationConverter):
    __provider__ = 'unicode_character_recognition'
    annotation_types = (CharacterRecognitionAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'images_dir': PathField(
                    is_directory=True, optional=True,
                    description='path to dataset images, used only for content existence check'
                ),
                'decoding_char_file': PathField(description='path to decoding_char_file')
            }
        )
        return parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir')
        self.decoding_char_file = self.get_value_from_config('decoding_char_file')
        self.supported_symbols = self.read_decoding_char_file(self.decoding_char_file, encoding='utf-8')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        content_errors = None
        if check_content:
            content_errors = []
            self.images_dir = self.images_dir or self.annotation_file.parent

        original_annotations = read_txt(self.annotation_file, encoding='utf-8')
        num_iterations = len(original_annotations)

        for line_id, line in enumerate(original_annotations):
            identifier, text = line.strip().split(',')
            annotations.append(CharacterRecognitionAnnotation(identifier.strip(), text.strip()))
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(identifier))
            if progress_callback is not None and line_id % progress_interval:
                progress_callback(line_id / num_iterations * 100)
        # index 0 is reserved for blank
        label_map = {ind+1: key for ind, key in enumerate(self.supported_symbols)}
        meta = {'label_map': label_map, 'blank_label': 0}
        return ConverterReturn(annotations, meta, content_errors)

    @staticmethod
    def read_decoding_char_file(file: Union[str, Path], **kwargs):
        with get_path(file).open(**kwargs) as content:
            lines = content.readlines()
            total_symbol = []
            for line in lines:
                total_symbol.append(line.strip())
            supported_symbols = ''.join(total_symbol)
            return supported_symbols


class KondateNakayosiRecognitionDatasetConverter(UnicodeCharacterRecognitionDatasetConverter):
    __provider__ = 'kondate_nakayosi_recognition'
