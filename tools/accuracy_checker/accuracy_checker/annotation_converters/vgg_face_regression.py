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

from ..config import PathField
from ..representation import FacialLandmarksAnnotation
from ..utils import convert_bboxes_xywh_to_x1y1x2y2, read_csv, check_file_existence

from .format_converter import BaseFormatConverter, ConverterReturn


class VGGFaceRegressionConverter(BaseFormatConverter):
    __provider__ = 'vgg_face'
    annotation_types = (FacialLandmarksAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'landmarks_csv_file': PathField(description="Path to csv file with coordinates of landmarks points."),
            'bbox_csv_file': PathField(
                optional=True, description="Path to cvs file which contains bounding box coordinates for faces."
            ),
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            )
        })

        return configuration_parameters

    def configure(self):
        self.landmarks_csv = self.get_value_from_config('landmarks_csv_file')
        self.bbox_csv = self.get_value_from_config('bbox_csv_file')
        self.images_dir = self.get_value_from_config('images_dir') or self.landmarks_csv.parent

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        content_errors = [] if check_content else None
        landmarks_table = read_csv(self.landmarks_csv)
        num_iterations = len(landmarks_table)
        for row_id, row in enumerate(landmarks_table):
            identifier = row['NAME_ID'] + '.jpg'
            x_values = np.array(
                [float(row["P1X"]), float(row["P2X"]), float(row["P3X"]), float(row["P4X"]), float(row["P5X"])]
            )
            y_values = np.array(
                [float(row["P1Y"]), float(row["P2Y"]), float(row["P3Y"]), float(row["P4Y"]), float(row["P5Y"])]
            )

            annotation = FacialLandmarksAnnotation(identifier, x_values, y_values)
            annotation.metadata['left_eye'] = 0
            annotation.metadata['right_eye'] = 1
            annotations.append(annotation)
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))

            if progress_callback and row_id % progress_interval == 0:
                progress_callback(row_id / num_iterations * 100)

        if self.bbox_csv:
            for index, row in enumerate(read_csv(self.bbox_csv)):
                annotations[index].metadata['rect'] = convert_bboxes_xywh_to_x1y1x2y2(
                    max(int(row["X"]), 0), max(int(row["Y"]), 0), max(int(row["W"]), 0), max(int(row["H"]), 0)
                )

        meta = {
            'label_map': {0: 'Left Eye', 1: 'Right Eye', 2: 'Nose', 3: 'Left Mouth Corner', 4: 'Right Mouth Corner'}
        }

        return ConverterReturn(annotations, meta, content_errors)
