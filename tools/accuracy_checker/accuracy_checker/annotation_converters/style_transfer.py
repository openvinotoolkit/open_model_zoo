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

from ..config import PathField
from ..representation import StyleTransferAnnotation
from .format_converter import BaseFormatConverter, ConverterReturn
from ..utils import read_txt, check_file_existence


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
            ),
            'annotation_file': PathField(optional=True, description='File with used images declaration')
        })
        return parameters

    def configure(self):
        self.image_dir = self.get_value_from_config('images_dir')
        self.annotation_file = self.get_value_from_config('annotation_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_check_errors = [] if check_content else None
        annotations = []
        if self.annotation_file:
            return self._convert_using_annotation(check_content, progress_callback, progress_interval)
        images = [
            im for im in self.image_dir.iterdir()
            if im.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
        ]
        num_iteration = len(images)
        for idx, image in enumerate(images):
            identifiers = image.name
            annotation = StyleTransferAnnotation(identifiers, image.name)
            annotations.append(annotation)
            if progress_callback and idx % progress_interval:
                progress_callback(idx * 100 / num_iteration)

        return ConverterReturn(annotations, None, content_check_errors)

    def _convert_using_annotation(self, check_content=False, progress_callback=None, progress_interval=100):
        list_images = read_txt(self.annotation_file)
        num_lines = len(list_images)
        annotation = []
        content_errors = [] if check_content else None
        for line_id, line in enumerate(list_images):
            data = line.split()
            if len(data) == 1:
                identifier = data[0]
                reference = data[0]
            elif len(data) == 2:
                identifier = data[0]
                reference = data[1]
            else:
                identifier = data[:-1]
                reference = data[-1]
            if check_content:
                if isinstance(identifier, str):
                    if not check_file_existence(self.image_dir / identifier):
                        content_errors.append(f'{self.image_dir / identifier}: does not exists')
                else:
                    for img in identifier:
                        if not check_file_existence(self.image_dir / img):
                            content_errors.append(f'{self.image_dir / identifier}: does not exists')
                if not check_file_existence(self.image_dir / reference):
                    content_errors.append(f'{self.image_dir/reference}: does not exist')
            ann = StyleTransferAnnotation(identifier, reference)
            if progress_callback and line_id % progress_interval:
                progress_callback(line_id * 100 / num_lines)
            annotation.append(ann)
        return ConverterReturn(annotation, None, content_errors)
