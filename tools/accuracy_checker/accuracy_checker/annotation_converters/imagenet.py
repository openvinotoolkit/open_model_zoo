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

import numpy as np

from ..config import PathField, BoolField
from ..representation import ClassificationAnnotation
from ..utils import read_txt, get_path, check_file_existence

from ..topology_types import ImageClassification
from .format_converter import BaseFormatConverter, ConverterReturn


class ImageNetFormatConverter(BaseFormatConverter):
    __provider__ = 'imagenet'
    annotation_types = (ClassificationAnnotation, )
    topology_types = (ImageClassification, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
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
            )
        })
        return parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.labels_file = self.get_value_from_config('labels_file')
        self.has_background = self.get_value_from_config('has_background')
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent

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

        meta = self._create_meta(self.labels_file, self.has_background) if self.labels_file else None

        return ConverterReturn(annotation, meta, content_errors)

    @staticmethod
    def _create_meta(labels_file, has_background=False):
        meta = {}
        labels = {}
        for i, line in enumerate(read_txt(get_path(labels_file))):
            index_for_label = i if not has_background else i + 1
            line = line.strip()
            label = line[line.find(' ') + 1:]
            labels[index_for_label] = label

        if has_background:
            labels[0] = 'background'
            meta['backgound_label'] = 0

        meta['label_map'] = labels

        return meta
