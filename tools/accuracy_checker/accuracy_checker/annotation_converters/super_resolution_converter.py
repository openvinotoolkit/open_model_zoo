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

import cv2
from ..config import PathField, StringField, BoolField, ConfigError, NumberField
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
                optional=True, default="lr", description="Low resolution file name's suffix."
            ),
            'hr_suffix': StringField(
                optional=True, default="hr", description="High resolution file name's suffix."
            ),
            'two_streams': BoolField(optional=True, default=False, description="2 streams is used"),
            'annotation_loader': StringField(
                optional=True, choices=LOADERS_MAPPING.keys(), default='pillow',
                description="Which library will be used for ground truth image reading. "
                            "Supported: {}".format(', '.join(LOADERS_MAPPING.keys()))
            ),
            'upsample_suffix': StringField(
                optional=True, default='upsample', description='Upsampled file name`s suffix, if 2 streams used.'
            ),
            'generate_upsample': BoolField(
                optional=True, default=False, description='allows to generate bicubic interpolation for images'
            ),
            'upsample_factor': NumberField(
                optional=True, default=4, value_type=int, min_value=1,
                description='upsampling factor if generation upsample used.'
            )
        })

        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.lr_suffix = self.get_value_from_config('lr_suffix')
        self.hr_suffix = self.get_value_from_config('hr_suffix')
        self.upsample_suffix = self.get_value_from_config('upsample_suffix')
        self.two_streams = self.get_value_from_config('two_streams')
        self.annotation_loader = LOADERS_MAPPING.get(self.get_value_from_config('annotation_loader'))
        if not self.annotation_loader:
            raise ConfigError('provided not existing loader')
        self.generate_upsample = self.get_value_from_config('generate_upsample')
        self.upsample_factor = self.get_value_from_config('upsample_factor')

    def convert(self):
        file_list_lr = []
        for file_in_dir in self.data_dir.iterdir():
            if self.lr_suffix in file_in_dir.parts[-1]:
                file_list_lr.append(file_in_dir)

        annotation = []
        for lr_file in file_list_lr:
            lr_file_name = lr_file.parts[-1]
            upsampled_file_name = self.upsample_suffix.join(lr_file_name.split(self.lr_suffix))
            if self.two_streams and self.generate_upsample:
                self.generate_upsample_file(lr_file, self.upsample_factor, upsampled_file_name)

            hr_file_name = self.hr_suffix.join(lr_file_name.split(self.lr_suffix))
            identifier = [lr_file_name, upsampled_file_name] if self.two_streams else lr_file_name
            annotation.append(SuperResolutionAnnotation(identifier, hr_file_name, gt_loader=self.annotation_loader))

        return annotation

    @staticmethod
    def generate_upsample_file(original_image_path, scale_factor, upsampled_file_name):
        image = cv2.imread(str(original_image_path))
        upsampled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(original_image_path.parent / upsampled_file_name), upsampled_image)
