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

import numpy as np

from ..config import PathField
from ..representation import FacialLandmarksAnnotation
from ..utils import read_txt, check_file_existence

from .format_converter import BaseFormatConverter, ConverterReturn


class WFLWConverter(BaseFormatConverter):
    __provider__ = 'wflw'
    annotation_types = (FacialLandmarksAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'annotation_file': PathField(
                description="Path to txt file, which contains ground truth data in WFLW dataset format."
            ),
            'images_dir': PathField(
                is_directory=True, optional=True,
                description="Path to dataset images, used only for content existence check."
            )
        })

        return configuration_parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.images_dir = self.get_value_from_config('images_dir')
        self.landmarks_count = 98
        self.coordinates_count = 2

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        image_annotations = read_txt(self.annotation_file)
        content_errors = None if not check_content else []
        annotations = []
        num_iterations = len(image_annotations)

        for index, line in enumerate(image_annotations):
            line = line.strip().split()
            identifier = line[-1]

            landmarks = [float(val) for val in line[:self.landmarks_count * self.coordinates_count]]
            x_values = np.array(landmarks[::2])
            y_values = np.array(landmarks[1::2])

            annotation = FacialLandmarksAnnotation(identifier, x_values, y_values)
            annotation.metadata['left_eye'] = [68, 72]
            annotation.metadata['right_eye'] = [60, 64]

            annotations.append(annotation)
            if check_content and self.images_dir:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))

            if progress_callback and index % progress_interval == 0:
                progress_callback(index * 100 / num_iterations)

        return ConverterReturn(annotations, None, content_errors)
