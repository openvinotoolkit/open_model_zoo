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
from ..utils import read_txt, check_file_existence
from ..config import PathField, BoolField
from ..representation import DetectionAnnotation


class CommonDetectionConverter(BaseFormatConverter):
    __provider__ = 'common_object_detection'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'images_dir': PathField(optional=True, is_directory=True, description='Images directory'),
            'annotation_dir': PathField(optional=False, is_directory=True, description='Annotation directory'),
            'labels_file': PathField(description='Labels file'),
            'pairs_file': PathField(
                optional=True, description='matching between images and annotations'
            ),
            'has_background': BoolField(optional=True, default=False, description='Indicator of background'),
            'add_background_to_label_id': BoolField(
                optional=True, default=False, description='Indicator that need shift labels'
            )
        })
        return params

    def configure(self):
        self.images_dir = self.get_value_from_config('images_dir')
        self.annotation_dir = self.get_value_from_config('annotation_dir')
        self.labels_file = self.get_value_from_config('labels_file')
        self.has_background = self.get_value_from_config('has_background')
        self.shift_labels = self.get_value_from_config('add_background_to_label_id')
        self.pairs_file = self.get_value_from_config('pairs_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_errors = None if not check_content else []
        image_ann_pairs = []
        if self.pairs_file:
            pairs_list = read_txt(self.pairs_file)
            for line in pairs_list:
                image_path, annotation_path = line.split(' ')
                image_path = image_path.split('@')[-1]
                annotation_path = annotation_path.split('@')[-1]
                image_ann_pairs.append((image_path, annotation_path))
        else:
            annotation_files = self.annotation_dir.glob('*.txt')
            for ann_file in annotation_files:
                image_ann_pairs.append((ann_file.name.replace('txt', 'jpg'), ann_file.name))
        num_iterations = len(image_ann_pairs)
        annotations = []
        for idx, (identifier, annotation_file) in enumerate(image_ann_pairs):
            labels, x_mins, y_mins, x_maxs, y_maxs = self.parse_annotation(annotation_file)
            annotations.append(DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs))
            if check_content:
                if self.images_dir is None:
                    self.images_dir = self.annotation_dir.parent / 'images'
                check_file_existence(self.images_dir / identifier)
                content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, self.generate_meta(), content_errors)

    def parse_annotation(self, annotation_file):
        labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
        for line in read_txt(self.annotation_dir / annotation_file):
            label, x_min, y_min, x_max, y_max = line.split()
            labels.append(int(label) + self.shift_labels)
            x_mins.append(float(x_min))
            y_mins.append(float(y_min))
            x_maxs.append(float(x_max))
            y_maxs.append(float(y_max))
        return np.array(labels), np.array(x_mins), np.array(y_mins), np.array(x_maxs), np.array(y_maxs)

    def generate_meta(self):
        labels = read_txt(self.labels_file)
        label_map = {}
        for idx, label_name in enumerate(labels):
            label_map[idx + self.has_background] = label_name
        meta = {'label_map': label_map}
        if self.has_background:
            meta['label_map'][0] = 'background'
            meta['background_label'] = 0
        return meta
