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
from .format_converter import BaseFormatConverter, ConverterReturn
from ..utils import read_txt, check_file_existence, convert_xctr_yctr_w_h_to_x1y1x2y2
from ..config import PathField
from ..representation import DetectionAnnotation


class YOLOLabelingConverter(BaseFormatConverter):
    __provider__ = 'yolo_labeling'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'data_dir': PathField(is_directory=True, description='Directory with annotations and images'),
            'labels_file': PathField(optional=True, description='Labels file')
        })
        return params

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.labels_file = self.get_value_from_config('labels_file')
        self.max_label = 0

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = None if not check_content else []
        image_ann_pairs = []

        annotation_files = self.data_dir.glob('*.txt')
        for ann_file in annotation_files:
            image_ann_pairs.append((ann_file.name.replace('txt', 'jpg'), ann_file.name))

        num_iterations = len(image_ann_pairs)
        annotations = []
        for idx, (identifier, annotation_file) in enumerate(image_ann_pairs):
            labels, x_mins, y_mins, x_maxs, y_maxs = self.parse_annotation(annotation_file)
            max_label_itr = max(labels)
            annotations.append(DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs))
            if check_content:
                check_file_existence(self.data_dir / identifier)
                content_errors.append('{}: does not exist'.format(self.data_dir / identifier))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

            self.max_label = max(self.max_label, max_label_itr)

        meta = self.generate_meta()

        return ConverterReturn(annotations, meta, content_errors)

    def parse_annotation(self, annotation_file):
        labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
        for line in read_txt(self.data_dir / annotation_file):
            label, x, y, width, height = line.split()
            x_min, y_min, x_max, y_max = convert_xctr_yctr_w_h_to_x1y1x2y2(
                float(x), float(y), float(width), float(height)
            )
            labels.append(int(label))
            x_mins.append(x_min)
            y_mins.append(y_min)
            x_maxs.append(x_max)
            y_maxs.append(y_max)
        return np.array(labels), np.array(x_mins), np.array(y_mins), np.array(x_maxs), np.array(y_maxs)

    def generate_meta(self):
        labels = read_txt(self.labels_file) if self.labels_file else range(self.max_label)
        label_map = {}
        for idx, label_name in enumerate(labels):
            label_map[idx] = label_name
        return {'label_map': label_map}
