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

import html
import unicodedata
from .format_converter import BaseFormatConverter, ConverterReturn
from ..config import PathField, BoolField
from ..representation import CharacterRecognitionAnnotation
from ..utils import read_txt


class BenthamOCRDatasetConverter(BaseFormatConverter):
    __provider__ = 'bentham_lines'


    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'transcription_dir': PathField(is_directory=True, description='Transcriptions directory.'),
            'partition_file': PathField(description='Dataset split for conversion'),
            'normalize_text': BoolField(optional=True, default=False),
            'to_lower': BoolField(optional=True, default=False)
        })

        return parameters

    def configure(self):
        self.transcription_dir = self.get_value_from_config('transcription_dir')
        self.partition_file = self.get_value_from_config('partition_file')
        self.normalize_text = self.get_value_from_config('normalize_text')
        self.to_lower = self.get_value_from_config('to_lower')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        image_list = read_txt(self.partition_file)
        annotations = []
        for idx in image_list:
            transcription_file = '{}.txt'.format(idx)
            identifier = '{}.png'.format(idx)
            text = ' '.join(read_txt(self.transcription_dir / transcription_file))
            text = html.unescape(text).replace("<gap/>", "")
            if self.normalize_text:
                text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
                text = " ".join(text.split())
            if self.to_lower:
                text = text.lower()
            annotations.append(CharacterRecognitionAnnotation(identifier, text))
        return ConverterReturn(annotations, None, None)
