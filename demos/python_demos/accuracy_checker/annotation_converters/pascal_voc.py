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

from accuracy_checker.representation import DetectionAnnotation, SegmentationAnnotation
from accuracy_checker.representation.segmentation_representation import GTMaskLoader
from accuracy_checker.utils import check_exists
from .format_converter import BaseFormatConverter

try:
    import lxml.etree as ET
except ImportError:
    import xml.etree.cElementTree as ET

_VOC_CLASSES = ('__background__',  # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')

_SEGMENTATION_COLORS = (((0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                         (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                         (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                         (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                         (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                         (0, 64, 128)))

_VOC_NUM_CLASSES = len(_VOC_CLASSES)

_CLASS_TO_IND = dict(zip(_VOC_CLASSES, list(range(_VOC_NUM_CLASSES))))


class PascalVOCSegmentationConverter(BaseFormatConverter):
    __provider__ = 'voc_segmentation'

    def convert(self, devkit_dir):
        """

        Args:
            devkit_dir: path to VOC2012 devkit dir (e.g. VOCdevkit/VOC2012)

        """

        devkit_dir = Path(devkit_dir)
        check_exists(devkit_dir.as_posix())

        image_set_file = devkit_dir / 'ImageSets' / 'Segmentation' / 'val.txt'
        mask_dir = devkit_dir / 'SegmentationClass'
        image_dir = devkit_dir / 'JPEGImages'

        with image_set_file.open() as f:
            image_list = f.read().strip().split()
            annotation = [SegmentationAnnotation(
                (image_dir / "{}.jpg".format(image)).as_posix(),
                (mask_dir / "{}.png".format(image)).as_posix(),
                mask_loader=GTMaskLoader.SCIPY
            ) for image in image_list]

        meta = {
            'label_map': dict(enumerate(_VOC_CLASSES)),
            'background_label': 0,
            'segmentation_colors': _SEGMENTATION_COLORS
        }

        return annotation, meta


class PascalVOCDetectionConverter(BaseFormatConverter):
    __provider__ = "voc07"

    def convert(self, devkit_dir):
        """
        Args:
            devkit_dir: path to VOC2007 devkit dir (e.g. .../VOCdevkit/VOC2007)
        """
        devkit_dir = Path(devkit_dir)
        check_exists(devkit_dir.as_posix())

        annotation_directory = devkit_dir / 'Annotations'
        images_directory = devkit_dir / 'JPEGImages'
        self.image_root = images_directory.as_posix()

        check_exists(annotation_directory.as_posix())
        check_exists(images_directory.as_posix())

        detections = []

        image_set_file = devkit_dir / 'ImageSets' / 'Main' / 'test.txt'

        with image_set_file.open() as f:
            image_list = f.read().strip().split()

        for image in tqdm(image_list):
            file_path = annotation_directory / '{}.xml'.format(image)
            tree = ET.parse(file_path.as_posix())

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

                labels.append(_CLASS_TO_IND[entry.find('name').text])
                x_mins.append(float(bbox.find('xmin').text) - 1)
                y_mins.append(float(bbox.find('ymin').text) - 1)
                x_maxs.append(float(bbox.find('xmax').text) - 1)
                y_maxs.append(float(bbox.find('ymax').text) - 1)

            image_annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)
            image_annotation.metadata['difficult_boxes'] = difficult_indices

            detections.append(image_annotation)

        meta = {
            'label_map': dict(enumerate(_VOC_CLASSES)),
            'background_label': 0
        }

        return detections, meta
