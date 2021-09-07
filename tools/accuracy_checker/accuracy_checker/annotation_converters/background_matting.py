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

from ..config import PathField, StringField
from .format_converter import BaseFormatConverter, ConverterReturn
from ..representation import BackgroundMattingAnnotation


class BackgroundMattingConverter(BaseFormatConverter):
    __provider__ = 'background_matting'

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update(
            {
                'images_dir': PathField(description='path to input images directory', is_directory=True),
                'masks_dir': PathField(description='path to gt masks directory', is_directory=True),
                'image_prefix': StringField(optional=True, default='', description='prefix for images'),
                'mask_prefix': StringField(optional=True, default='', description='prefix for gt masks'),
                'image_postfix': StringField(optional=True, default='.png', description='prefix for images'),
                'mask_postfix': StringField(optional=True, default='.png', description='prefix for gt masks'),
            }
        )
        return configuration_parameters

    def configure(self):
        self.images_dir = self.get_value_from_config('images_dir')
        self.masks_dir = self.get_value_from_config('masks_dir')
        self.images_prefix = self.get_value_from_config('image_prefix')
        self.images_postfix = self.get_value_from_config('image_postfix')
        self.mask_prefix = self.get_value_from_config('mask_prefix')
        self.mask_postfix = self.get_value_from_config('mask_postfix')
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        mask_name = '{prefix}{base}{postfix}'.format(
            prefix=self.mask_prefix, base='{base}', postfix=self.mask_postfix
        )
        image_pattern = '*'
        if self.images_prefix:
            image_pattern = self.images_prefix + image_pattern
        if self.images_postfix:
            image_pattern = image_pattern + self.images_postfix
        images_list = list(self.images_dir.glob(image_pattern))
        num_iterations = len(images_list)
        content_errors = None if not check_content else []
        for idx, image in enumerate(images_list):
            base_name = image.name
            identifier = base_name
            if self.images_prefix:
                base_name = base_name.split(self.images_prefix)[-1]
            if self.images_postfix:
                base_name = base_name.split(self.images_postfix)[0]

            mask_file = self.masks_dir / mask_name.format(base=base_name)
            if not mask_file.exists():
                content_errors.append('{}: does not exist'.format(mask_file))

            annotations.append(
                BackgroundMattingAnnotation(identifier, mask_file.name)
            )
            if progress_callback is not None and idx % progress_interval == 0:
                progress_callback(idx / num_iterations * 100)

        return ConverterReturn(
            annotations, {'label_map': {'background': 0, 'foreground': list(range(1, 256))}}, content_errors
        )
