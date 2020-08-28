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

from ..config import PathField, BoolField
from ..representation import ClassificationAnnotation
from ..utils import read_txt, get_path, check_file_existence, read_json

from ..topology_types import ImageClassification
from .format_converter import BaseFormatConverter, ConverterReturn, verify_label_map


class ImageNetFormatConverter(BaseFormatConverter):
    __provider__ = 'imagenet'
    annotation_types = (ClassificationAnnotation, )
    topology_types = (ImageClassification, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'annotation_file': PathField(description="Path to annotation in txt format."),
            'labels_file': PathField(
                optional=True,
                description="Path to file with word description of labels (synset words)."
            ),
            'has_background': BoolField(
                optional=True, default=False,
                description="Allows to add background label to original labels and"
                            " convert dataset for 1001 classes instead 1000."
            ),
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            ),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding)', optional=True
            )
        })
        return configuration_parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.labels_file = self.get_value_from_config('labels_file')
        self.has_background = self.get_value_from_config('has_background')
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = []
        content_errors = [] if check_content else None
        original_annotation = read_txt(get_path(self.annotation_file))
        num_iterations = len(original_annotation)
        for image_id, image in enumerate(original_annotation):
            image_name, label = image.split()
            if check_content:
                if not check_file_existence(self.images_dir / image_name):
                    content_errors.append('{}: does not exist'.format(self.images_dir / image_name))

            label = np.int64(label) if not self.has_background else np.int64(label) + 1
            annotation.append(ClassificationAnnotation(image_name, label))
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id / num_iterations * 100)

        meta = self._create_meta(self.labels_file, self.dataset_meta, self.has_background) or None

        return ConverterReturn(annotation, meta, content_errors)

    @staticmethod
    def _create_meta(labels_file, dataset_meta, has_background=False):
        meta = {}
        label_map = {}
        if dataset_meta:
            meta = read_json(dataset_meta)
            if 'labels' in dataset_meta and 'label_map' not in meta:
                labels = ['background'] + meta['labels'] if has_background else meta['labels']
                label_map = dict(enumerate(labels))
                meta['label_map'] = label_map
            else:
                if 'label_map' in meta:
                    meta['label_map'] = verify_label_map(meta['label_map'])
                return meta

        if labels_file:
            label_map = {}
            for i, line in enumerate(read_txt(get_path(labels_file))):
                index_for_label = i if not has_background else i + 1
                line = line.strip()
                label = line[line.find(' ') + 1:]
                label_map[index_for_label] = label

            meta['label_map'] = label_map

        if has_background:
            label_map[0] = 'background'
            meta['background_label'] = 0

        return meta
