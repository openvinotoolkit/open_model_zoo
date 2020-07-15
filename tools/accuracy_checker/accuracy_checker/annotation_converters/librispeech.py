"""
Copyright (c) 2019 Intel Corporation

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

from pathlib import Path
import re

from ..representation import CharacterRecognitionAnnotation
from .format_converter import DirectoryBasedAnnotationConverter
from .format_converter import ConverterReturn

class LibrispeechConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'librispeech'
    annotation_types = (CharacterRecognitionAnnotation, )

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')

    def convert(self, check_content=False, **kwargs):

        pattern = re.compile(r'([0-9\-]+)\s+(.+)')
        annotations = []
        data_folder = Path(self.data_dir)
        txts = list(data_folder.glob('**/*.txt'))
        for txt in txts:
            content = open(txt).readlines()
            for line in content:
                res = pattern.search(line)
                if res:
                    name = res.group(1)
                    transcript = res.group(2)
                    fname = txt.parent / name
                    fname = fname.with_suffix('.wav')

                annotations.append(CharacterRecognitionAnnotation(str(fname.relative_to(data_folder)), transcript))

        return ConverterReturn(annotations, None, None)
