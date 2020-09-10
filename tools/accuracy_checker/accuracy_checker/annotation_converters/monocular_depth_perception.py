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

from .format_converter import BaseFormatConverter, ConverterReturn
from ..utils import get_path, check_file_existence, contains_any, read_txt
from ..config import PathField, ConfigError
from ..representation import DepthEstimationAnnotation


class ReDWebDatasetConverter(BaseFormatConverter):
    __provider__ = 'redweb'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update(
            {
                'data_dir': PathField(optional=True, is_directory=True, description='dataset root directory'),
                'annotation_file': PathField(
                    optional=True, description='txt file which represent a list of pairs of image and depth map')
            }
        )
        return params

    def configure(self):
        if not contains_any(self.config, ['data_dir', 'annotation_file']):
            raise ConfigError('data_dir or annotation_file should be provided')
        self.data_dir = self.get_value_from_config('data_dir')
        self.annotation_file = self.get_value_from_config('annotation_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        if self.annotation_file is not None:
            return self._convert_annotation_file_based(check_content, progress_callback, progress_interval)
        return self._convert_directory_based(check_content, progress_callback, progress_interval)

    def _convert_annotation_file_based(self, check_content=False, progress_callback=None, progress_interval=100):
        if self.data_dir is None:
            self.data_dir = self.annotation_file.parent
        content_errors = [] if check_content else None
        annotations = []
        list_of_pairs = read_txt(self.annotation_file)

        num_iterations = len(list_of_pairs)
        for idx, line in enumerate(list_of_pairs):
            identifier, depth_map_path = line.split(' ')
            if check_content:
                if not check_file_existence(self.data_dir / depth_map_path):
                    content_errors.append('{}: does not exists'.format(self.data_dir / depth_map_path))
                if not check_file_existence(self.data_dir / identifier):
                    content_errors.append('{}: does not exists'.format(self.data_dir / identifier))
            annotations.append(DepthEstimationAnnotation(identifier, depth_map_path))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, None, check_content)

    def _convert_directory_based(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        identifier_prefix = 'imgs'
        images_dir = get_path(self.data_dir / identifier_prefix, is_directory=True)
        relative_depth_prefix = 'RD'
        content_errors = [] if check_content else None
        images_list = list(images_dir.glob('*.jpg'))
        num_iterations = len(images_list)
        annotations = []
        for idx, image_path in enumerate(images_list):
            identifier = '{}/{}'.format(identifier_prefix, image_path.name)
            depth_map_file = image_path.name.split(image_path.suffix)[0] + '.png'
            depth_map_path = '{}/{}'.format(relative_depth_prefix, depth_map_file)
            if check_content and not check_file_existence(self.data_dir / depth_map_path):
                content_errors.append('{}: does not exists'.format(self.data_dir / depth_map_path))
            annotations.append(DepthEstimationAnnotation(identifier, depth_map_path))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, None, check_content)
