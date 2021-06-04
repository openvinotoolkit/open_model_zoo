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

from ..config import PathField, NumberField
from ..representation import DetectionAnnotation
from ..utils import read_csv, check_file_existence
from .format_converter import BaseFormatConverter, ConverterReturn


class OpenImagesDetectionConverter(BaseFormatConverter):
    __provider__ = 'open_images_detection'
    annotation_types = (DetectionAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'bbox_csv_file': PathField(description='Path to cvs file which contains bounding box coordinates'),
            'labels_file': PathField(description='Path to cvs file which contains class labels'),
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='Path to dataset images, used only for content existence check'
            ),
            'label_start': NumberField(
                value_type=int, optional=True, default=1,
                description='Specifies label index start in label map. Default value is 1. You can provide another'
                            'value, if you want to use this dataset for separate label validation.'
            )
        })

        return parameters

    def configure(self):
        self.bbox_csv = self.get_value_from_config('bbox_csv_file')
        self.labels_file = self.get_value_from_config('labels_file')
        self.images_dir = self.get_value_from_config('images_dir')
        self.label_start = self.get_value_from_config('label_start')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        content_errors = [] if check_content else None
        bboxes = read_csv(self.bbox_csv)
        label_map, annotations_label_map = self.get_labels_mapping(self.labels_file, self.label_start)
        annotation_by_identifier = {}
        for row in bboxes:
            if annotation_by_identifier.get(row['ImageID']):
                annotation_by_identifier[row['ImageID']].append(row)
            else:
                annotation_by_identifier[row['ImageID']] = [row]
        num_iterations = len(annotation_by_identifier)
        for idx, (identifier, data) in enumerate(annotation_by_identifier.items()):
            identifier = identifier + '.jpg'
            x_mins, y_mins, x_maxs, y_maxs, labels = [], [], [], [], []
            for row in data:
                x_mins.append(float(row['XMin']))
                x_maxs.append(float(row['XMax']))
                y_mins.append(float(row['YMin']))
                y_maxs.append(float(row['YMax']))
                labels.append(annotations_label_map[row['LabelName']])

            annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)
            annotations.append(annotation)
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))

            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx / num_iterations * 100)

        return ConverterReturn(annotations, {'label_map': label_map}, content_errors)

    @staticmethod
    def get_labels_mapping(labels_file, label_start):
        labels = read_csv(labels_file, is_dict=False)
        label_map = {}
        reversed_annotation_label_map = {}
        for idx, (label_name, real_name) in enumerate(labels, start=label_start):
            label_map[idx] = real_name
            reversed_annotation_label_map[label_name] = idx
        return label_map, reversed_annotation_label_map
