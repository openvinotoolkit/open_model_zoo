"""
Copyright (c) 2018-2022 Intel Corporation

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

from csv import DictReader
from ..config import PathField, NumberField
from ..utils import get_path, check_file_existence
from ..representation import SegmentationAnnotation
from .format_converter import BaseFormatConverter, ConverterReturn


class ADE20kConverter(BaseFormatConverter):
    __provider__ = 'ade20k'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'images_dir': PathField(is_directory=True, description='Images directory'),
            'annotations_dir': PathField(is_directory=True, description='Annotation directory'),
            'object_categories_file': PathField(description='file with object categories'),
            'num_classes': NumberField(description='Number of used classes', optional=True, value_type=int,
                                       min_value=2),
        })
        return parameters

    def configure(self):
        self.images_dir = self.get_value_from_config('images_dir')
        self.annotation_dir = self.get_value_from_config('annotations_dir')
        self.object_categories_file = self.get_value_from_config('object_categories_file')
        self.num_classes = self.get_value_from_config('num_classes')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = None if not check_content else []
        images_paths = list(get_path(self.images_dir, is_directory=True).glob('*.jpg'))
        num_iterations = len(images_paths)
        annotations = []
        for idx, image in enumerate(images_paths):
            identifier = image.name
            annotation_path = self.annotation_dir / identifier.replace('jpg', 'png')
            if check_content:
                if not check_file_existence(annotation_path):
                    content_errors.append('{}: does not exist'.format(annotation_path))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)
            annotations.append(SegmentationAnnotation(identifier, annotation_path.name))
        return ConverterReturn(annotations, self.get_meta(), content_errors)

    def get_meta(self):
        categories_dist = DictReader(self.object_categories_file.open(), delimiter='\t')
        if self.num_classes:
            label_map = {int(category['Idx']): category['Name'] for category in categories_dist
                         if int(category['Idx']) < self.num_classes}
        else:
            label_map = {int(category['Idx']): category['Name'] for category in categories_dist}
        label_map[0] = 'background'
        return {'label_map': label_map, 'background_label': 0}
