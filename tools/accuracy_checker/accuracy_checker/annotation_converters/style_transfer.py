"""
Copyright (c) 2020 Intel Corporation

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
from ..representation import StyleTransferAnnotation
from .format_converter import BaseFormatConverter, ConverterReturn


class StyleTransferConverter(BaseFormatConverter):
    __provider__ = 'style_transfer'
    annotation_types = (StyleTransferAnnotation,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'images_dir': PathField(
                optional=False, is_directory=True,
                description="Path to directory with images."
            )
        })
        return parameters

    def configure(self):
        self.image_dir = self.get_value_from_config('images_dir')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_check_errors = [] if check_content else None
        annotations = []
        images = list(im for im in self.image_dir.iterdir())
        for image in images:
            identifiers = image.name
            annotation = StyleTransferAnnotation(identifiers, image.name)
            annotations.append(annotation)

        return ConverterReturn(annotations, None,
                               content_check_errors)
