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
from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import FacialLandmarksAnnotation
from ..utils import read_xml, check_file_existence
from ..config import PathField


class CVATFacialLandmarksRecognitionConverter(FileBasedAnnotationConverter):
    __provider__ = 'cvat_facial_landmarks'
    annotation_types = (FacialLandmarksAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            )
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = read_xml(self.annotation_file)
        meta = annotation.find('meta')
        size = int(meta.find('task').find('size').text)
        num_landmarks = len(list(meta.iter('label')))
        annotations = []
        content_errors = None if not check_content else []
        for image_id, image in enumerate(annotation.iter('image')):
            identifier = image.attrib['name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            landmarks_x, landmarks_y = self.get_landmarks(image, num_landmarks)
            landmarks_annotation = FacialLandmarksAnnotation(identifier, np.array(landmarks_x), np.array(landmarks_y))
            landmarks_annotation.metadata['left_eye'] = [1, 0]
            landmarks_annotation.metadata['right_eye'] = [2, 3]
            annotations.append(landmarks_annotation)
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id * 100 / size)

        return ConverterReturn(annotations, None, content_errors)

    @staticmethod
    def get_landmarks(image_annotation, num_landmarks):
        landmarks_x, landmarks_y = np.zeros(num_landmarks), np.zeros(num_landmarks)
        for point in image_annotation:
            idx = int(point.attrib['label'])
            x, y = point.attrib['points'].split(',')
            landmarks_x[idx] = float(x)
            landmarks_y[idx] = float(y)

        return landmarks_x, landmarks_y
