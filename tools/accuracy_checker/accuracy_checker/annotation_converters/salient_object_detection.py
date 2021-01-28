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

from .format_converter import BaseFormatConverter, ConverterReturn
from ..config import PathField
from ..representation import SalientRegionAnnotation
from ..utils import check_file_existence, read_txt


class SalientObjectDetectionConverter(BaseFormatConverter):
    __provider__ = 'salient_object_detection'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'images_dir': PathField(is_directory=True, description='directory with images'),
            'masks_dir': PathField(is_directory=True, description='directory with salient region mask'),
            'annotation_file': PathField(optional=True, description='target image id list')
        })
        return params

    def configure(self):
        self.images_dir = self.get_value_from_config('images_dir')
        self.masks_dir = self.get_value_from_config('masks_dir')
        self.annotation_file = self.get_value_from_config('annotation_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = [] if check_content else None
        annotations = []
        if self.annotation_file:
            image_ids = ['{}.jpg'.format(im_id) for im_id in read_txt(self.annotation_file)]
        else:
            image_ids = [image.name for image in self.images_dir.glob('*.jpg')]
        num_iterations = len(image_ids)
        for idx, identifier in enumerate(image_ids):
            map_identifier = identifier.replace('jpg', 'png')
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
                if not check_file_existence(self.masks_dir / map_identifier):
                    content_errors.append('{}: does not exist'.format(self.masks_dir / map_identifier))
            if progress_callback is not None and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)
            annotations.append(SalientRegionAnnotation(identifier, map_identifier))

        return ConverterReturn(annotations, None, content_errors)
