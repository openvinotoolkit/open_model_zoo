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

from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import DetectionAnnotation
from ..topology_types import ObjectDetection
from ..utils import read_xml, check_file_existence, read_json
from ..config import PathField, ConfigError, BoolField


class CVATObjectDetectionConverter(FileBasedAnnotationConverter):
    __provider__ = 'cvat_object_detection'
    annotation_types = (DetectionAnnotation, )
    topology_types = (ObjectDetection, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            ),
            'has_background': BoolField(optional=True, default=True, description='Dataset has background label or not'),
            'labels_file': PathField(optional=True, description='path to label map in json format'),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding)', optional=True
            )
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.has_background = self.get_value_from_config('has_background')
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.label_map_file = self.get_value_from_config('labels_file')
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = read_xml(self.annotation_file)
        annotation_meta = annotation.find('meta')
        size = int(annotation_meta.find('task').find('size').text)
        label_to_id, meta = self.generate_labels_mapping(annotation_meta)
        annotations = []
        content_errors = None if not check_content else []
        for image_id, image in enumerate(annotation.iter('image')):
            identifier = image.attrib['name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            x_mins, y_mins, x_maxs, y_maxs, labels_ids, difficult = [], [], [], [], [], []
            for _, bbox in enumerate(image):
                if 'label' not in bbox.attrib.keys() or bbox.attrib['label'] not in label_to_id:
                    continue
                labels_ids.append(label_to_id[bbox.attrib['label']])
                x_mins.append(float(bbox.attrib['xtl']))
                y_mins.append(float(bbox.attrib['ytl']))
                x_maxs.append(float(bbox.attrib['xbr']))
                y_maxs.append(float(bbox.attrib['ybr']))
                if 'occluded' in bbox.attrib and int(bbox.attrib['occluded']):
                    difficult.append(len(labels_ids) - 1)
            detection_annotation = DetectionAnnotation(identifier, labels_ids, x_mins, y_mins, x_maxs, y_maxs)
            detection_annotation.metadata['difficult_boxes'] = difficult
            annotations.append(detection_annotation)
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id * 100 / size)

        return ConverterReturn(annotations, meta, content_errors)

    def generate_labels_mapping(self, annotation_meta):
        if self.dataset_meta:
            meta = read_json(self.dataset_meta)
            if 'labels' in meta and 'label_map' not in meta:
                offset = int(self.has_background)
                label_to_id = {label_name: label_id + offset for label_id, label_name in enumerate(meta['labels'])}
                meta['label_map'] = {'label_map': {value: key for key, value in label_to_id.items()}}
                if self.has_background:
                    meta['label_map'][0] = 'background'
                    meta['background_label'] = 0

            label_map = meta.get('label_map')
            if not label_map:
                raise ConfigError('dataset_meta_file should contains labels or label_map')
            label_to_id = {value: key for key, value in label_map.items()}

            return label_to_id, meta

        meta = {}
        if self.label_map_file:
            label_to_id = read_json(self.label_map_file).get('labels')
            if not label_to_id:
                raise ConfigError('label_map_file does not contains labels key')
        else:
            labels = [label.find('name').text for label in annotation_meta.iter('label') if label.find('name').text]
            if not labels:
                raise ConfigError('annotation file does not contains labels')
            if self.has_background:
                labels = ['background'] + labels
                meta['background_label'] = 0
            label_to_id = {label: idx for idx, label in enumerate(labels)}
        meta['label_map'] = {value: key for key, value in label_to_id.items()}

        return label_to_id, meta
