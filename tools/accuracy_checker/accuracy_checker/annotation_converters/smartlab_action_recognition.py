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

from .format_converter import BaseFormatConverter, ConverterReturn
from ..config import NumberField, PathField
from ..representation import ClassificationAnnotation
from ..utils import read_txt, get_path


class SmartLabActionRecognition(BaseFormatConverter):
    __provider__ = 'smartlab_action_recognition'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'stream': NumberField(optional=False, description='currently used stream id', value_type=int),
            'labels_dir': PathField(is_directory=True, optional=False, description='directory with label files')
        })
        return params

    def configure(self):
        self.stream_id = self.get_value_from_config('stream')
        self.labels_dir = self.get_value_from_config('labels_dir')
        self.stream_file = get_path(self.labels_dir / 'streams_{}.txt'.format(self.stream_id))

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        stream_description = read_txt(self.stream_file)
        num_iterations = len(stream_description)
        for idx, annotation_line in enumerate(stream_description):
            identifier, label = annotation_line.split()
            label = int(label) if int(label) in [1, 2] else 0
            annotations.append(ClassificationAnnotation(identifier, label))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)
        return ConverterReturn(annotations, self.get_meta(), None)

    def get_meta(self):
        return {'label_map': {0: 'no_action', 1: 'noise_action', 2: 'adjust_rider'}}
