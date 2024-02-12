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

from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..utils import read_txt
from ..representation import CharacterRecognitionAnnotation


class CommonTextRecognition(FileBasedAnnotationConverter):
    __provider__ = 'common_text_recognition'

    def convert(self, check_content=False, **kwargs):
        annotations = []
        for line in read_txt(self.annotation_file):
            input_file, description = line.split(' ', 1)
            annotations.append(CharacterRecognitionAnnotation(input_file, description))
        return ConverterReturn(annotations, None, None)
