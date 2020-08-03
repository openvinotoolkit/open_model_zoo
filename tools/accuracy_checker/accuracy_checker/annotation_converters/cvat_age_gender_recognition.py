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

from ..utils import read_xml, check_file_existence, contains_all
from ..config import PathField, ConfigError
from ..representation import ContainerAnnotation, ClassificationAnnotation, RegressionAnnotation
from .format_converter import FileBasedAnnotationConverter, ConverterReturn


class CVATAgeGenderRecognitionConverter(FileBasedAnnotationConverter):
    __provider__ = 'cvat_age_gender'
    annotation_types = (ClassificationAnnotation, RegressionAnnotation, )

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
        target_label = self.select_label(meta).find('name').text
        gender_classes_mapping = {'female': 0, 'male': 1}
        meta = {
            'age_label_map': {0: 'child', 1: 'young', 2: 'middle', 3: 'old'},
            'gender_label_map': {0: 'female', 1: 'male'}
        }

        annotations = []
        content_errors = None if not check_content else []
        for image_id, image in enumerate(annotation.iter('image')):
            identifier = image.attrib['name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            for bbox in image:
                if 'label' not in bbox.attrib.keys() or bbox.attrib['label'] != target_label:
                    continue
                annotation_dict = {}
                bbox_rect = [
                    float(bbox.attrib['xtl']), float(bbox.attrib['ytl']),
                    float(bbox.attrib['xbr']), float(bbox.attrib['ybr'])
                ]
                for attribute in bbox.iter('attribute'):
                    attribute_name = attribute.attrib['name']
                    if attribute_name == 'gender':
                        attribute_label = gender_classes_mapping[attribute.text]
                        attribute_annotation = ClassificationAnnotation(identifier, attribute_label)
                        attribute_annotation.metadata['rect'] = bbox_rect
                        annotation_dict['gender_annotation'] = attribute_annotation
                    if attribute_name == 'age':
                        attribute_value = int(attribute.text)
                        attribute_label = self.get_age_class(attribute_value)
                        attribute_class_annotation = ClassificationAnnotation(identifier, attribute_label)
                        attribute_regression_annotation = RegressionAnnotation(identifier, attribute_value)
                        attribute_class_annotation.metadata['rect'] = bbox_rect
                        attribute_regression_annotation.metadata['rect'] = bbox_rect
                        annotation_dict['age_class_annotation'] = attribute_class_annotation
                        annotation_dict['age_regression_annotation'] = attribute_regression_annotation
                annotations.append(ContainerAnnotation(annotation_dict))
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id * 100 / size)

        return ConverterReturn(annotations, meta, content_errors)

    @staticmethod
    def get_age_class(age):
        if age < 19:
            return 0
        if age < 36:
            return 1
        if age < 66:
            return 2

        return 3

    @staticmethod
    def select_label(meta):
        def check_contains_attributes(label):
            attribute_names = [attribute.find('name').text for attribute in label.iter('attribute')]
            return contains_all(attribute_names, ['age', 'gender'])
        label = [label for label in meta.iter('label') if check_contains_attributes(label)]
        if not label:
            raise ConfigError('annotation does not contains label with age and gender attributes')

        return label[0]
