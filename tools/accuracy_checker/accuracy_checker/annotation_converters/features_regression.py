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

from ..config import PathField, StringField, BoolField
from ..representation import FeaturesRegressionAnnotation
from ..utils import check_file_existence
from .format_converter import ConverterReturn, BaseFormatConverter


class FeaturesRegressionConverter(BaseFormatConverter):
    __provider__ = 'feature_regression'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'input_dir': PathField(is_directory=True, description='directory with input data'),
            'reference_dir': PathField(is_directory=True, description='directory for reference files storing'),
            'input_suffix': StringField(optional=True, default='.txt', description='suffix for input file'),
            'reference_suffix': StringField(optional=True, default='.txt', description='suffix for reference file'),
            'use_bin_data': BoolField(optional=True, default=False, description='use binary data formats'),
            'bin_data_dtype': StringField(optional=True, default='float32', description='binary data type for reading')
        })
        return params

    def configure(self):
        self.in_directory = self.get_value_from_config('input_dir')
        self.ref_directory = self.get_value_from_config('reference_dir')
        self.in_suffix = self.get_value_from_config('input_suffix')
        self.ref_suffix = self.get_value_from_config('reference_suffix')
        self.bin_data = self.get_value_from_config('use_bin_data')
        self.bin_dtype = self.get_value_from_config('bin_data_dtype')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        ref_data_list = list(self.ref_directory.glob('*{}'.format(self.ref_suffix)))
        for ref_file in ref_data_list:
            identifier = ref_file.name.replace(self.ref_suffix, self.in_suffix)
            input_file = self.in_directory / identifier
            if not input_file.exists():
                continue
            annotations.append(
                FeaturesRegressionAnnotation(identifier, ref_file.name, is_bin=self.bin_data, bin_dtype=self.bin_dtype)
            )
        return ConverterReturn(annotations, None, None)


class MultiOutputFeaturesRegression(BaseFormatConverter):
    __provider__ = 'multi_feature_regression'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'data_dir': PathField(is_directory=True, description='directory with data'),
            'input_suffix': StringField(optional=True, default='in.npy', description='suffix for input files search'),
            'reference_suffix': StringField(optional=True, default='out.npy', description='suffix for ref files'),
            'prefix': StringField(optional=True, default='', description='prefix for files search')
        })
        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.input_suffix = self.get_value_from_config('input_suffix')
        self.output_suffix = self.get_value_from_config('reference_suffix')
        self.prefix = self.get_value_from_config('prefix')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        input_data = list(self.data_dir.glob('{}*{}*'.format(self.prefix, self.input_suffix)))
        content_errors = None if not check_content else []
        num_iterations = len(input_data)
        for idx, input_file in enumerate(input_data):
            ref_file = input_file.parent / input_file.name.replace(self.input_suffix, self.output_suffix)
            if check_content and not check_file_existence(ref_file):
                content_errors.append('{}: does not exist'.format(ref_file))
            annotations.append(FeaturesRegressionAnnotation(input_file.name, ref_file.name, dict_features=True))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)
        return ConverterReturn(annotations, None, None)
