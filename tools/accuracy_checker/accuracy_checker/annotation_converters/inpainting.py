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
from ..logging import warning
from ..representation import ImageInpaintingAnnotation
from .format_converter import BaseFormatConverter, ConverterReturn


class InpaintingConverter(BaseFormatConverter):
    __provider__ = 'inpainting'
    annotation_types = (ImageInpaintingAnnotation,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'images_dir': PathField(
                optional=False, is_directory=True,
                description="Path to directory with images."
            ),
            'masks_dir': PathField(
                optional=True, is_directory=True,
                description="Path to mask dataset. If not specified masks will be generated automatically."
            )
        })

        return parameters

    def configure(self):
        self.image_dir = self.get_value_from_config('images_dir')
        self.masks_dir = self.get_value_from_config('masks_dir')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_check_errors = [] if check_content else None

        annotations = []
        images = list(im for im in self.image_dir.iterdir())
        if self.masks_dir is not None:
            masks = list(mask for mask in self.masks_dir.iterdir())
            if len(masks) < len(images):
                warning('Number of masks is smaller than number of images.'
                        'Only {} first images will be used'.format(len(masks)))
                images = images[:len(masks)]
            if len(images) < len(masks):
                warning('Number of images is smaller than number of masks.'
                        'Only {} first masks will be used'.format(len(images)))
                masks = masks[:len(images)]

        for i, image in enumerate(images):
            mask_name = None if self.masks_dir is None else masks[i].name
            identifier = image.name
            annotation = ImageInpaintingAnnotation([identifier], identifier)
            annotation.metadata['mask'] = {
                'mask_name': mask_name
            }
            annotations.append(annotation)

        return ConverterReturn(annotations, None, content_check_errors)
