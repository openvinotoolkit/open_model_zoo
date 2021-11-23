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

from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import DetectionAnnotation, ActionDetectionAnnotation, ContainerAnnotation
from ..utils import read_xml, check_file_existence
from ..config import PathField, ConfigError, StringField


ACTIONS = {
    'common_3_actions': {
        'listening': 0,
        'reading': 0,
        'writing': 0,
        'lie_on_the_desk': 0,
        'sitting': 0,
        'standing': 1,
        'raising_hand': 2,
        '__undefined__': 3
    },

    'common_6_actions': {
        'sitting': 0,
        'reading': 0,
        'writing': 1,
        'raising_hand': 2,
        'standing': 3,
        'turned_around': 4,
        'lie_on_the_desk': 5,
        '__undefined__': 6
    },

    'teacher': {
        'standing': 0,
        'speaking': 0,
        'writing': 1,
        'demonstrating': 2,
        'interacting': 0,
        '__undefined__': 3
    },

    'raising_hand': {
        'listening': 0,
        'reading': 0,
        'writing': 0,
        'lie_on_the_desk': 0,
        'sitting': 0,
        'standing': 0,
        'raising_hand': 1,
        '__undefined__': 2
    }
}
ACTIONS_BACK = {
    'common_3_actions': {
        0: 'sitting',
        1: 'standing',
        2: 'raising_hand'
    },

    'common_6_actions': {
        0: 'sitting',
        1: 'writing',
        2: 'raising_hand',
        3: 'standing',
        4: 'turned_around',
        5: 'lie_on_the_desk',
    },

    'teacher': {
        0: 'standing',
        1: 'writing',
        2: 'demonstrating',
    },

    'raising_hand': {
        'seating': 0,
        'raising_hand': 1
    }
}


class CVATPersonDetectionActionRecognitionConverter(FileBasedAnnotationConverter):
    __provider__ = 'cvat_person_detection_action_recognition'
    annotation_types = (DetectionAnnotation, ActionDetectionAnnotation)

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            ),
            'use_case': StringField(
                optional=True, choices=ACTIONS.keys(), default='common_3_actions',
                description="Use case, which determine the dataset label map. "
                            "Supported range actions: {}".format(', '.join(ACTIONS.keys()))
            ),
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.use_case = self.get_value_from_config('use_case')
        self.action_names_map = ACTIONS.get(self.use_case)
        self.action_names_back_map = ACTIONS_BACK.get(self.use_case)

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = read_xml(self.annotation_file)
        meta = annotation.find('meta')
        size = int(meta.find('task').find('size').text)
        label = [label.find('name').text for label in meta.iter('label') if label.find('name').text == 'person']
        if not label:
            raise ConfigError('annotation file does not contains person label')

        annotations = []
        content_errors = None if not check_content else []
        for image_id, image in enumerate(annotation.iter('image')):
            identifier = image.attrib['name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            x_mins, y_mins, x_maxs, y_maxs, labels_ids, difficult = [], [], [], [], [], []
            for bbox_id, bbox in enumerate(image):
                if 'label' not in bbox.attrib.keys() or bbox.attrib['label'] != 'person':
                    continue
                action = [attribute for attribute in bbox.iter('attribute') if attribute.attrib['name'] == 'action']
                if not action:
                    continue
                action = action[0].text
                if action not in self.action_names_map:
                    continue
                labels_ids.append(self.action_names_map[action])
                x_mins.append(float(bbox.attrib['xtl']))
                y_mins.append(float(bbox.attrib['ytl']))
                x_maxs.append(float(bbox.attrib['xbr']))
                y_maxs.append(float(bbox.attrib['ybr']))
                if 'occluded' in bbox.attrib and int(bbox.attrib['occluded']):
                    difficult.append(bbox_id)
            detection_annotation = DetectionAnnotation(identifier, [1]*len(x_mins), x_mins, y_mins, x_maxs, y_maxs)
            action_annotation = ActionDetectionAnnotation(identifier, labels_ids, x_mins, y_mins, x_maxs, y_maxs)
            detection_annotation.metadata['difficult_boxes'] = difficult
            action_annotation.metadata['difficult_boxes'] = difficult
            annotations.append(ContainerAnnotation({
                'person_annotation': detection_annotation, 'action_annotation': action_annotation
            }))
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id * 100 / size)

        meta = {
            'action_label_map': self.action_names_back_map,
            'person_label_map': {1: 'person'}
        }

        return ConverterReturn(annotations, meta, content_errors)
