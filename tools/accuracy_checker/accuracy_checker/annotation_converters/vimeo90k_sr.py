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

from .format_converter import BaseFormatConverter, ConverterReturn
from ..data_readers import MultiFramesInputIdentifier
from ..config import PathField, BoolField
from ..representation import SuperResolutionAnnotation
from ..utils import read_txt


class Vimeo90KSuperResolutionDatasetConverter(BaseFormatConverter):
    __provider__ = 'vimeo90k'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'annotation_file': PathField(description='testing split file'),
            'add_flow': BoolField(optional=True, default=False)
        })
        return params

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.add_flow = self.get_value_from_config('add_flow')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        test_set = read_txt(self.annotation_file)
        annotations = []
        for sept in test_set:
            target = 'target/{}/im4.png'.format(sept)
            input_frame = 'low_resolution/{}/im4.png'.format(sept)
            neibhors = ['low_resolution/{}/im{}.png'.format(sept, idx) for idx in range(1, 8) if idx != 4]
            input_data = [input_frame] + neibhors
            if self.add_flow:
                input_data += ['flow/{}/flow_{}.npy'.format(sept, idx) for idx in range(6)]
            annotations.append(SuperResolutionAnnotation(
                MultiFramesInputIdentifier(list(range(len(input_data))), input_data), target))
        return ConverterReturn(annotations, None, None)
