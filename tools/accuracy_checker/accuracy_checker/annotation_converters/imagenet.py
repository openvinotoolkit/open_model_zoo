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
from ..utils import read_txt, get_path

from ..topology_types import ImageClassification
from .format_converter import BaseFormatConverter

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
            )
        })
        return parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.labels_file = self.get_value_from_config('labels_file')
        self.has_background = self.get_value_from_config('has_background')

    def convert(self):
        annotation = []
        for image in read_txt(get_path(self.annotation_file)):
            image_name, label = image.split()
            label = np.int64(label) if not self.has_background else np.int64(label) + 1
            annotation.append(ClassificationAnnotation(image_name, label))
        meta = self._create_meta(self.labels_file, self.has_background) if self.labels_file else None

        return annotation, meta

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
