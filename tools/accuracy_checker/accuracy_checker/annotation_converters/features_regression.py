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

from ..config import PathField
from ..representation import FeaturesRegressionAnnotation
from .format_converter import ConverterReturn, BaseFormatConverter, StringField


class FeaturesRegressionConverter(BaseFormatConverter):
    __provider__ = 'feature_regression'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'input_dir': PathField(is_directory=True),
            'reference_dir': PathField(is_directory=True),
            'input_suffix': StringField(optional=True, default='.txt'),
            'reference_suffix': StringField(optional=True, default='.txt')
        })
        return params

    def configure(self):
        self.in_directory = self.get_value_from_config('input_dir')
        self.ref_directory = self.get_value_from_config('reference_dir')
        self.in_suffix = self.get_value_from_config('input_suffix')
        self.ref_suffix = self.get_value_from_config('reference_suffix')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        ref_data_list = list(self.ref_directory.glob('*{}'.format(self.ref_suffix)))
        for ref_file in ref_data_list:
            identifier = ref_file.name.replace(self.ref_suffix, self.in_suffix)
            input_file = self.in_directory / identifier
            if not input_file.exists():
                continue
            annotations.append(FeaturesRegressionAnnotation(identifier, ref_file.name))
        return ConverterReturn(annotations, None, None)
