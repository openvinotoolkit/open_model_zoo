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
from ..representation import MultiLabelRecognitionAnnotation
from ..utils import read_xml, check_file_existence
from ..config import StringField, PathField, ConfigError


class CVATMultilabelAttributesRecognitionConverter(FileBasedAnnotationConverter):
    __provider__ = 'cvat_multilabel_binary_attributes_recognition'
    annotation_types = (MultiLabelRecognitionAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'label': StringField(description='specific label for attribute collection'),
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            )
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.label = self.get_value_from_config('label')
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = read_xml(self.annotation_file)
        meta = annotation.find('meta')
        size = int(meta.find('task').find('size').text)
        label = self.select_label(meta)
        label_to_id = {attribute.find('name').text: idx for idx, attribute in enumerate(label.iter('attribute'))}
        num_attributes = len(label_to_id)

        annotations = []
        content_errors = None if not check_content else []
        for image_id, image in enumerate(annotation.iter('image')):
            identifier = image.attrib['name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            for bbox in image:
                if 'label' not in bbox.attrib.keys() or bbox.attrib['label'] != self.label:
                    continue
                bbox_rect = [
                    float(bbox.attrib['xtl']), float(bbox.attrib['ytl']),
                    float(bbox.attrib['xbr']), float(bbox.attrib['ybr'])
                ]
                attributes = -np.ones(num_attributes)
                for attribute in bbox.iter('attribute'):
                    attribute_name = attribute.attrib['name']
                    attribute_label = label_to_id[attribute_name]
                    attributes[attribute_label] = 1 if attribute.text == 'T' else 0
                attributes_annotation = MultiLabelRecognitionAnnotation(identifier, attributes)
                attributes_annotation.metadata['rect'] = bbox_rect
                annotations.append(attributes_annotation)

                if progress_callback is not None and image_id % progress_interval == 0:
                    progress_callback(image_id * 100 / size)

        return ConverterReturn(annotations, self.generate_meta(label_to_id), content_errors)

    @staticmethod
    def generate_meta(attribute_values_mapping):
        return {'label_map': {value: key for key, value in attribute_values_mapping.items()}}

    def select_label(self, meta):
        label = [label for label in meta.iter('label') if label.find('name').text == self.label]
        if not label:
            raise ConfigError('{} does not present in annotation'.format(self.label))
        return label[0]
