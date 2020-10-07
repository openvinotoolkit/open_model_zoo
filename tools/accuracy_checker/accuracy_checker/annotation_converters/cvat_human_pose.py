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

import numpy as np
from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import PoseEstimationAnnotation
from ..utils import read_xml, check_file_existence
from ..config import PathField

LABELS_TO_COCO = {
    'nose': 0,
    'r_shoulder': 6,
    'r_elbow': 8,
    'r_wrist': 9,
    'l_shoulder': 5,
    'l_elbow': 7,
    'l_wrist': 10,
    'r_hip': 12,
    'r_knee': 14,
    'r_ankle': 16,
    'l_hip': 11,
    'l_knee': 13,
    'l_ankle': 15,
    'r_eye': 2,
    'l_eye': 1,
    'r_ear': 3,
    'l_ear': 4
}


class CVATPoseEstimationConverter(FileBasedAnnotationConverter):
    __provider__ = 'cvat_pose_estimation'
    annotation_types = (PoseEstimationAnnotation, )

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
        annotations = []
        content_errors = None if not check_content else []
        for image_id, image in enumerate(annotation.iter('image')):
            identifier = image.attrib['name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            label = [1]
            x_vals, y_vals = np.zeros((1, len(LABELS_TO_COCO))), np.zeros((1, len(LABELS_TO_COCO)))
            visilibity = np.zeros((1, len(LABELS_TO_COCO)))
            for point in image.iter('points'):
                point_label = point.attrib['label']
                if point_label not in LABELS_TO_COCO:
                    continue
                point_id = LABELS_TO_COCO[point_label]
                point_x, point_y = point.attrib['points'].split(',')
                x_vals[0, point_id] = float(point_x)
                y_vals[0, point_id] = float(point_y)
                if int(point.attrib['occluded']):
                    continue
                visilibity[0, point_id] = 2
            annotations.append(PoseEstimationAnnotation(identifier, x_vals, y_vals, visilibity, label))

            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id * 100 / size)
        meta = {'label_map': {1: 'person'}}

        return ConverterReturn(annotations, meta, content_errors)

    @staticmethod
    def get_pose(image_annotation, num_landmarks):
        landmarks_x, landmarks_y = np.zeros(num_landmarks), np.zeros(num_landmarks)
        for point in image_annotation:
            idx = int(point.attrib['label'])
            x, y = point.attrib['points'].split(',')
            landmarks_x[idx] = float(x)
            landmarks_y[idx] = float(y)

        return landmarks_x, landmarks_y
