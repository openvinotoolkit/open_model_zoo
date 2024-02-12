"""
Copyright (c) 2018-2024 Intel Corporation

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
from ..utils import read_txt, check_file_existence
from ..config import PathField, NumberField, StringField
from ..representation import DetectionAnnotation


class KITTIConverter(BaseFormatConverter):
    __provider__ = 'kitti_2d_detection'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'images_dir': PathField(optional=True, is_directory=True, description='Images directory'),
            'annotation_dir': PathField(optional=False, is_directory=True, description='Annotation directory'),
            'labels_file': PathField(description='Labels file'),
            'label_start': NumberField(
                value_type=int, optional=True, default=1,
                description='Specifies label index start in label map. Default value is 1. You can provide another'
                            'value, if you want to use this dataset for separate label validation.'
            ),
            'images_suffix': StringField(optional=True, default='.png', description='Suffix for images'),
        })
        return params

    def configure(self):
        self.images_dir = self.get_value_from_config('images_dir')
        self.annotation_dir = self.get_value_from_config('annotation_dir')
        self.labels_file = self.get_value_from_config('labels_file')
        self.label_start = self.get_value_from_config('label_start')
        self.images_suffix = self.get_value_from_config('images_suffix')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = None if not check_content else []
        ann_files = self.annotation_dir.glob('*.txt')
        image_ann_pairs = [(ann.name.replace('.txt', self.images_suffix), ann.name) for ann in ann_files]

        meta = self.get_meta()
        self.reversed_label_map = {value: key for key, value in meta['label_map'].items()}
        num_iterations = len(image_ann_pairs)
        annotations = []
        for idx, (identifier, annotation_files) in enumerate(image_ann_pairs):
            labels, x_mins, y_mins, x_maxs, y_maxs = self.parse_annotation(annotation_files)
            annotations.append(DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs))
            if check_content:
                if self.images_dir is None:
                    self.images_dir = self.annotation_dir.parent / 'image_2'
                check_file_existence(self.images_dir / identifier)
                content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, meta, content_errors)

    def parse_annotation(self, annotation_file):
        labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
        for line in read_txt(self.annotation_dir / annotation_file):
            annotation_line = line.split()
            label = annotation_line[0].lower()
            if not self.reversed_label_map.get(label):
                continue
            x_min, y_min, x_max, y_max = annotation_line[4:8]
            labels.append(self.reversed_label_map[label])
            x_mins.append(float(x_min))
            y_mins.append(float(y_min))
            x_maxs.append(float(x_max))
            y_maxs.append(float(y_max))
        return np.array(labels), np.array(x_mins), np.array(y_mins), np.array(x_maxs), np.array(y_maxs)

    def get_meta(self):
        labels = read_txt(self.labels_file)
        label_map = {}
        for idx, label_name in enumerate(labels, start=self.label_start):
            label_map[idx] = label_name.lower()
        meta = {'label_map': label_map}
        return meta
