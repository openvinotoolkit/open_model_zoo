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

from pathlib import Path
import re
import json
import numpy as np

from ..representation import CharacterRecognitionAnnotation
from ..config import PathField, NumberField, BoolField
from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn


class LibrispeechConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'librispeech'
    annotation_types = (CharacterRecognitionAnnotation, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'annotation_file': PathField(optional=True),
            'top_n': NumberField(optional=True, value_type=int),
            'use_numpy': BoolField(optional=True, default=False)
        })
        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.top_n = self.get_value_from_config('top_n')
        self.numpy_files = self.get_value_from_config('use_numpy')

    def convert(self, check_content=False, **kwargs):
        _, file_list = self.create_annotation_list()
        pattern = re.compile(r'([0-9\-]+)\s+(.+)')
        annotations = []
        data_folder = Path(self.data_dir)
        txts = list(data_folder.glob('**/*.txt'))
        for txt in txts:
            content = txt.open().readlines()
            for line in content:
                res = pattern.search(line)
                if res:
                    name = res.group(1)
                    transcript = res.group(2)
                    fname = txt.parent / name
                    fname = fname.with_suffix('.wav' if not self.numpy_files else '.npy')
                    if file_list and fname.name not in file_list:
                        continue
                    identifier = str(fname.relative_to(data_folder))
                    annotations.append(CharacterRecognitionAnnotation(
                        identifier, transcript.upper()
                    ))
        return ConverterReturn(annotations, None, None)

    def create_annotation_list(self):
        annotation_list = []
        durations = []
        file_names = []
        if self.annotation_file is None:
            return [], []
        with self.annotation_file.open() as json_file:
            for line in json_file:
                record = json.loads(line)
                annotation_list.append(record)
                durations.append(float(record['duration']))
                filename = Path(record["audio_filepath"]).name
                if self.numpy_files:
                    filename = filename.replace('wav', 'npy')
                file_names.append(filename)

        if self.top_n:
            sorted_by_duration = np.argsort(durations)
            subset = sorted_by_duration[:self.top_n] if len(sorted_by_duration) > self.top_n else sorted_by_duration
            return [annotation_list[idx] for idx in subset], [file_names[idx] for idx in subset]
        return annotation_list, file_names
