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
from .format_converter import ConverterReturn, BaseFormatConverter
from ..config import PathField, ListField, StringField, ConfigError
from ..representation import ImageProcessingAnnotation
from .image_processing import LOADERS_MAPPING
from ..utils import get_path


class GANAnnotationConverter(BaseFormatConverter):
    __provider__ = 'image_generation'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'data_dir': PathField(description='dataset root directory', is_directory=True),
            'input_subdirectories': ListField(value_type=str, allow_empty=False,
                                              description='subdirectories with input data'),
            'reference_dir': PathField(
                is_directory=True,
                description='path to directory with reference relative to dataset root'
            ),
            'annotation_loader': StringField(
                optional=True, choices=LOADERS_MAPPING.keys(), default='pillow',
                description="Which library will be used for ground truth image reading. "
                            "Supported: {}".format(', '.join(LOADERS_MAPPING.keys()))
            )
        })
        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.input_subdir = [
            get_path(self.data_dir / path, is_directory=True)
            for path in self.get_value_from_config('input_subdirectories')
        ]
        self.reference_dir = self.get_value_from_config('reference_dir')
        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        if not self.annotation_loader:
            raise ConfigError('provided not existing loader')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        for ref_file in self.reference_dir.glob('*'):
            input_files = []
            for input_path in self.input_subdir:
                input_candidate = list(input_path.glob('{}.*'.format(ref_file.stem)))
                if not input_candidate:
                    raise ConfigError('Input data for {} is not found'.format(ref_file))
                input_files.append(str(input_candidate[0].relative_to(self.data_dir)))
            identifier = input_files if len(input_files) > 1 else input_files[0]
            ref_path = ref_file.name
            annotations.append(ImageProcessingAnnotation(identifier, ref_path, gt_loader=self.annotation_loader))
        return ConverterReturn(annotations, None, None)
