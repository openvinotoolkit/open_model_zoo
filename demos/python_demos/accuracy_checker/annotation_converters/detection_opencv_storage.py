"""
Copyright (c) 2018 Intel Corporation

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

from accuracy_checker.representation import DetectionAnnotation
from accuracy_checker.utils import convert_bboxes_xywh_to_x1y1x2y2, read_xml, read_txt

from .format_converter import BaseFormatConverter


class DetectionOpenCVStorageFormatConverter(BaseFormatConverter):
    __provider__ = "detection_opencv_storage"

    def convert(self, file_path, image_names=None, label_start=1, background_label=None):
        root = read_xml(file_path)

        labels_set = self.get_label_set(root)

        label_start = int(label_start)
        labels_set = sorted(labels_set)
        class_to_ind = dict(zip(labels_set, list(range(label_start, len(labels_set) + label_start + 1))))
        label_map = {}
        for class_label, ind in class_to_ind.items():
            label_map[ind] = class_label

        annotations = []
        for frames in root:
            for frame in frames:
                identifier = frame.tag + '.png'
                labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
                difficult_indices = []
                for annotation in frame:
                    label = annotation.find('type')
                    if label is None:
                        raise ValueError('"{}" contains detection without "{}"'.format(file_path, 'type'))

                    box = annotation.find('roi')
                    if box is None:
                        raise ValueError('"{}" contains detection without "{}"'.format(file_path, 'roi'))
                    box = list(map(float, box.text.split()))

                    is_ignored = annotation.find('is_ignored')
                    if is_ignored is not None and int(is_ignored.text) == 1:
                        difficult_indices.append(len(labels))

                    labels.append(class_to_ind[label.text])
                    x_min, y_min, x_max, y_max = convert_bboxes_xywh_to_x1y1x2y2(*box)
                    x_mins.append(x_min)
                    y_mins.append(y_min)
                    x_maxs.append(x_max)
                    y_maxs.append(y_max)

                detection_annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)
                detection_annotation.metadata['difficult_boxes'] = difficult_indices
                annotations.append(detection_annotation)

        if image_names:
            self.rename_identifiers(annotations, image_names)

        meta = {}
        if background_label:
            self.add_background(label_map, meta, background_label)
        meta['label_map'] = label_map

        return annotations, meta

    @staticmethod
    def rename_identifiers(annotation_list, images_file: str):
        for annotation, image in zip(annotation_list, read_txt(images_file)):
            annotation.identifier = image

        return annotation_list

    @staticmethod
    def add_background(label_map, meta, background_label):
        background_label = int(background_label)
        label_map[background_label] = '__background__'
        meta['background_label'] = background_label

    @staticmethod
    def get_label_set(xml_root):
        labels_set = set()
        for frames in xml_root:
            for frame in frames:
                for annotation in frame:
                    label = annotation.find('type')
                    if label is None:
                        raise ValueError('annotation contains detection without label')

                    labels_set.add(label.text)

        return labels_set
