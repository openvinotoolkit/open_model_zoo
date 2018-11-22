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
from pathlib import Path
from accuracy_checker.representation import DetectionAnnotation
from accuracy_checker.utils import check_exists

from .format_converter import BaseFormatConverter

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.cElementTree as ET


class DetectionOpenCVStorageFormatConverter(BaseFormatConverter):
    __provider__ = "detection_opencv_storage"

    def convert(self, file_path, image_names=None, label_start=1, background_label=None):
        """
        Args:
            file_path: path to file with data
        """
        check_exists(file_path)
        label_start = int(label_start)

        tree = ET.parse(file_path)
        labels_set = set()
        for frames in tree.getroot():
            for frame in frames:
                for annotation in frame:
                    label = annotation.find('type')
                    if label is None:
                        raise ValueError('"{}" contains detection without label'.format(file_path))

                    labels_set.add(label.text)

        labels_set = sorted(labels_set)
        class_to_ind = dict(zip(labels_set, list(range(label_start, len(labels_set) + label_start + 1))))
        label_map = {}
        for class_label, ind in class_to_ind.items():
            label_map[ind] = class_label

        annotations = []
        for frames in tree.getroot():
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
                    x_mins.append(box[0])
                    y_mins.append(box[1])
                    x_maxs.append(box[0] + box[2])
                    y_maxs.append(box[1] + box[3])

                detection_annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)
                detection_annotation.metadata['difficult_boxes'] = difficult_indices
                annotations.append(detection_annotation)

        if image_names is not None:
            self.rename_identifiers(annotations, image_names)
        meta = {}
        self.add_background(label_map, meta, background_label)
        meta['label_map'] = label_map

        return annotations, meta

    @staticmethod
    def rename_identifiers(annotation_list, images_file):
        check_exists(images_file)
        with Path(images_file).open() as images_f:
            images_list = images_f.read().split('\n')
        for annotation, image in zip(annotation_list, images_list):
            annotation.identifier = image
        return annotation_list

    @staticmethod
    def add_background(label_map, meta, background_label=None):
        if background_label is not None:
            background_label = int(background_label)
            label_map[background_label] = '__background__'
            meta['background_label'] = background_label
