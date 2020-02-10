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

from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn
from ..utils import get_path, check_file_existence
from ..representation import DepthEstimationAnnotation


class ReDWebDatasetConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'redweb'

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        images_list = []

        with open(get_path(self.data_dir / 'ReDWeb_validation_360.txt')) as f:
            for i, row in enumerate(f):
                row = row.replace("\r", "").replace("\n", "")
                parts = row.split(" ")

                images_list.append(get_path(self.data_dir / parts[0]))

                # TODO: remove (used for debugging)
                if i == 2:
                    break

        relative_depth_prefix = 'RDs'
        content_errors = [] if check_content else None
        num_iterations = len(images_list)

        annotations = []
        for idx, image_path in enumerate(images_list):
            identifier = '{}/{}'.format("Imgs", image_path.name)
            depth_map_file = image_path.name.split(image_path.suffix)[0] + '.png'
            depth_map_path = '{}/{}'.format(relative_depth_prefix, depth_map_file)
            if check_content and not check_file_existence(self.data_dir / depth_map_path):
                content_errors.append('{}: does not exists'.format(self.data_dir / depth_map_path))
            annotations.append(DepthEstimationAnnotation(identifier, depth_map_path))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, None, check_content)
        