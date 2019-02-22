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

import errno
import os
from pathlib import Path
from tqdm import tqdm

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.cElementTree as ET

from accuracy_checker.representation import DetectionAnnotation, SegmentationAnnotation
from accuracy_checker.representation.segmentation_representation import GTMaskLoader
from accuracy_checker.utils import get_path, string_to_bool, read_txt
from .format_converter import BaseFormatConverter

_VOC_CLASSES_DETECTION = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

_VOC_CLASSES_SEGMENTATION = tuple(['__background__']) + _VOC_CLASSES_DETECTION
_SEGMENTATION_COLORS = ((
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
    (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128)
))


def prepare_detection_labels(has_background=True):
    num_classes = len(_VOC_CLASSES_DETECTION)
    labels_shift = 1 if has_background else 0
    reversed_label_map = dict(zip(_VOC_CLASSES_DETECTION, list(range(labels_shift, num_classes + labels_shift))))
    if has_background:
        reversed_label_map['__background__'] = 0
    return reversed_label_map


def reverse_label_map(label_map):
    return {value: key for key, value in label_map.items()}


class PascalVOCSegmentationConverter(BaseFormatConverter):
    __provider__ = 'voc_segmentation'

    def convert(self, devkit_dir):
        """
        Args:
            devkit_dir: path to VOC2012 devkit dir (e.g. VOCdevkit/VOC2012)
        """
        devkit_dir = get_path(devkit_dir, is_directory=True)

        image_set_file = devkit_dir / 'ImageSets' / 'Segmentation' / 'test.txt'
        mask_dir = Path('SegmentationClass')
        image_dir = Path('JPEGImages')

        annotations = []
        for image in read_txt(image_set_file):
            annotation = SegmentationAnnotation(
                str(image_dir / '{}.jpg'.format(image)),
                str(mask_dir / '{}.png'.format(image)),
                mask_loader=GTMaskLoader.SCIPY
            )

            annotations.append(annotation)

        meta = {
            'label_map': dict(enumerate(_VOC_CLASSES_SEGMENTATION)),
            'background_label': 0,
            'segmentation_colors': _SEGMENTATION_COLORS
        }

        return annotations, meta


class PascalVOCDetectionConverter(BaseFormatConverter):
    __provider__ = "voc07"

    def convert(self, devkit_dir, has_background=True):
        """
        Args:
            devkit_dir: path to VOC2007 devkit dir (e.g. .../VOCdevkit/VOC2007)
            has_background: allows to add background label to label map
        """
        if isinstance(has_background, str):
            has_background = string_to_bool(has_background)

        class_to_ind = prepare_detection_labels(has_background)
        devkit_dir = get_path(devkit_dir, is_directory=True)

        annotation_directory = get_path(devkit_dir / 'Annotations', is_directory=True)
        images_directory = get_path(devkit_dir / 'JPEGImages', is_directory=True)

        detections = []
        image_set_file = devkit_dir / 'ImageSets' / 'Main' / 'test.txt'
        for image in tqdm(read_txt(image_set_file, sep=None)):
            file_path = annotation_directory / '{}.xml'.format(image)
            tree = ET.parse(str(file_path))

            identifier = tree.find('.//filename').text
            image_path = images_directory / identifier

            if not image_path.is_file():
                raise FileNotFoundError("{}: {}".format(os.strerror(errno.ENOENT), image_path))

            labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
            difficult_indices = []
            for entry in tree.getroot():
                if not entry.tag.startswith('object'):
                    continue

                bbox = entry.find('bndbox')
                difficult = int(entry.find('difficult').text)

                if difficult == 1:
                    difficult_indices.append(len(labels))

                labels.append(class_to_ind[entry.find('name').text])
                x_mins.append(float(bbox.find('xmin').text) - 1)
                y_mins.append(float(bbox.find('ymin').text) - 1)
                x_maxs.append(float(bbox.find('xmax').text) - 1)
                y_maxs.append(float(bbox.find('ymax').text) - 1)

            image_annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)
            image_annotation.metadata['difficult_boxes'] = difficult_indices

            detections.append(image_annotation)

        meta = {'label_map': reverse_label_map(class_to_ind)}
        if has_background:
            meta['background_label'] = 0

        return detections, meta
