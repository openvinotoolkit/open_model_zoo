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

from ..config import PathField, StringField, BoolField, ConfigError
from ..representation import SuperResolutionAnnotation
from ..representation.super_resolution_representation import GTLoader
from .format_converter import BaseFormatConverter

LOADERS_MAPPING = {
    'opencv': GTLoader.OPENCV,
    'pillow': GTLoader.PILLOW
}

class SRConverter(BaseFormatConverter):
    __provider__ = 'super_resolution'
    annotation_types = (SuperResolutionAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'data_dir': PathField(
                is_directory=True, description="Path to folder, where images in low and high resolution are located."
            ),
            'lr_suffix': StringField(
                optional=True, default="lr", description="Low resolution file name's suffix."),
            'hr_suffix': StringField(
                optional=True, default="hr", description="High resolution file name's suffix."
            ),
            'two_streams': BoolField(optional=True, default=False, description="2 streams is used"),
            'annotation_loader': StringField(
                optional=True, choices=LOADERS_MAPPING.keys(), default='pillow',
                description="Which library will be used for ground truth image reading. "
                            "Supported: {}".format(', '.join(LOADERS_MAPPING.keys())))
        })

        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.lr_suffix = self.get_value_from_config('lr_suffix')
        self.hr_suffix = self.get_value_from_config('hr_suffix')
        self.two_streams = self.get_value_from_config('two_streams')
        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        if not self.annotation_loader:
            raise ConfigError('provided not existing loader')

    def convert(self):
        file_list_lr = []
        for file_in_dir in self.data_dir.iterdir():
            if self.lr_suffix in file_in_dir.parts[-1]:
                file_list_lr.append(file_in_dir)

        annotation = []
        for lr_file in file_list_lr:
            lr_file_name = lr_file.parts[-1]
            hr_file_name = self.hr_suffix.join(lr_file_name.split(self.lr_suffix))
            identifier = [lr_file_name, hr_file_name] if self.two_streams else lr_file_name
            annotation.append(SuperResolutionAnnotation(identifier, hr_file_name, gt_loader=self.annotation_loader))

        return annotation
