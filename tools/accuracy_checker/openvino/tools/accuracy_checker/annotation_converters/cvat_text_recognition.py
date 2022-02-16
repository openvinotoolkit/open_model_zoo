"""
Copyright (c) 2018-2022 Intel Corporation

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

from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import CharacterRecognitionAnnotation
from ..utils import read_xml, check_file_existence
from ..config import PathField, ConfigError


class CVATTextRecognitionConverter(FileBasedAnnotationConverter):
    __provider__ = 'cvat_text_recognition'
    annotation_types = (CharacterRecognitionAnnotation, )
    supported_symbols = '0123456789abcdefghijklmnopqrstuvwxyz'

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            ),
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
                bbox_rect = [
                    float(bbox.attrib['xtl']), float(bbox.attrib['ytl']),
                    float(bbox.attrib['xbr']), float(bbox.attrib['ybr'])
                ]
                for attribute in bbox.iter('attribute'):
                    attribute_name = attribute.attrib['name']
                    if attribute_name != 'text':
                        continue
                    text_recognition_annotation = CharacterRecognitionAnnotation(identifier, attribute.text)
                    text_recognition_annotation.metadata['rect'] = bbox_rect
                    annotations.append(text_recognition_annotation)
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id * 100 / size)

        return ConverterReturn(annotations, self.get_meta(), content_errors)

    def get_meta(self):
        label_map = {ind: str(key) for ind, key in enumerate(self.supported_symbols)}
        return {'label_map': label_map, 'blank_label': len(label_map)}

    @staticmethod
    def select_label(meta):
        def check_contains_attributes(label):
            attribute_names = [attribute.find('name').text for attribute in label.iter('attribute')]
            return 'text' in attribute_names
        label = [label for label in meta.iter('label') if check_contains_attributes(label)]
        if not label:
            raise ConfigError('annotation does not contains label with text attribute')

        return label[0]
