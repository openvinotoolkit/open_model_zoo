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

from ..config import PathField, NumberField, ConfigError
from ..representation import DetectionAnnotation
from ..utils import convert_bboxes_xywh_to_x1y1x2y2, read_xml, read_txt, check_file_existence, read_json

from .format_converter import BaseFormatConverter, ConverterReturn, verify_label_map


class DetectionOpenCVStorageFormatConverter(BaseFormatConverter):
    __provider__ = 'detection_opencv_storage'
    annotation_types = (DetectionAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'annotation_file': PathField(description="Path to annotation in xml format."),
            'image_names_file': PathField(
                optional=True,
                description="Path to txt file, which contains image name list for dataset."
            ),
            'label_start': NumberField(
                value_type=int, optional=True, default=1,
                description="Specifies label index start in label map. Default value is 1. "
                            "You can provide another value, if you want to use this dataset "
                            "for separate label validation."
            ),
            'background_label': NumberField(
                value_type=int, optional=True,
                description="Specifies which index will be used for background label. "
                            "You can not provide this parameter if your dataset has not background label."
            ),
            'data_dir': PathField(
                is_directory=True, optional=True,
                description='this parameter used only for dataset image existence validation purposes.'
            ),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding)', optional=True
            )
        })

        return configuration_parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.image_names_file = self.get_value_from_config('image_names_file')
        self.label_start = self.get_value_from_config('label_start')
        self.background_label = self.get_value_from_config('background_label')
        self.data_dir = self.get_value_from_config('data_dir')
        if self.data_dir is None:
            self.data_dir = self.annotation_file.parent
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        def update_progress(frame_id):
            if progress_callback is not None and frame_id % progress_interval == 0:
                progress_callback(frame_id / num_iterations * 100)

        root = read_xml(self.annotation_file)
        class_to_ind, meta = self.generate_meta(root)

        content_check_errors = None

        annotations = []
        for frames in root:
            num_iterations = len(frames)
            for frame_id, frame in enumerate(frames):
                identifier = '{}.png'.format(frame.tag)
                labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
                difficult_indices = []
                for annotation in frame:
                    label = annotation.findtext('type')
                    if not label:
                        raise ValueError('"{}" contains detection without "{}"'.format(self.annotation_file, 'type'))

                    box = annotation.findtext('roi')
                    if not box:
                        raise ValueError('"{}" contains detection without "{}"'.format(self.annotation_file, 'roi'))
                    box = list(map(float, box.split()))

                    is_ignored = annotation.findtext('is_ignored', 0)
                    if int(is_ignored) == 1:
                        difficult_indices.append(len(labels))

                    labels.append(class_to_ind[label])
                    x_min, y_min, x_max, y_max = convert_bboxes_xywh_to_x1y1x2y2(*box)
                    x_mins.append(x_min)
                    y_mins.append(y_min)
                    x_maxs.append(x_max)
                    y_maxs.append(y_max)

                detection_annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)
                detection_annotation.metadata['difficult_boxes'] = difficult_indices
                annotations.append(detection_annotation)
                update_progress(frame_id)

        if self.image_names_file:
            self.rename_identifiers(annotations, self.image_names_file)

        if check_content:
            content_check_errors = []
            for annotation in annotations:
                if not check_file_existence(self.data_dir / annotation.identifier):
                    content_check_errors.append('{}: file not found'.format(self.data_dir / annotation.identifier))

        return ConverterReturn(annotations, meta, content_check_errors)

    @staticmethod
    def rename_identifiers(annotation_list, images_file):
        for annotation, image in zip(annotation_list, read_txt(images_file)):
            annotation.identifier = image

        return annotation_list

    @staticmethod
    def get_label_set(xml_root):
        labels_set = set()
        for frames in xml_root:
            for frame in frames:
                for annotation in frame:
                    label = annotation.findtext('type')
                    if not label:
                        raise ValueError('annotation contains detection without label')

                    labels_set.add(label)

        return labels_set

    def generate_meta(self, root):
        if self.dataset_meta:
            meta = read_json(self.dataset_meta)
            if 'labels' in meta and 'label_map' not in meta:
                labels_set = meta['labels']
                class_to_ind = dict(
                    zip(labels_set, list(range(self.label_start, len(labels_set) + self.label_start + 1)))
                )
                meta['label_map'] = {'label_map': {value: key for key, value in class_to_ind.items()}}
                if self.background_label:
                    meta['label_map'][self.background_label] = '__background__'
                    meta['background_label'] = 0
            label_map = meta.get('label_map')
            if not label_map:
                raise ConfigError('dataset_meta_file should contains labels or label_map')
            label_map = verify_label_map(label_map)
            class_to_ind = {value: key for key, value in label_map.items()}

            return class_to_ind, meta

        labels_set = self.get_label_set(root)
        labels_set = sorted(labels_set)
        class_to_ind = dict(zip(labels_set, list(range(self.label_start, len(labels_set) + self.label_start + 1))))
        label_map = {}
        for class_label, ind in class_to_ind.items():
            label_map[ind] = class_label
        meta = {}
        if self.background_label:
            label_map[self.background_label] = '__background__'
            meta['background_label'] = self.background_label
        meta['label_map'] = label_map

        return class_to_ind, meta
