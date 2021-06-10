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

from ..config import PathField, BoolField
from ..representation import DetectionAnnotation, SegmentationAnnotation
from ..representation.segmentation_representation import GTMaskLoader
from ..utils import get_path, read_txt, read_xml, check_file_existence, read_json
from .format_converter import BaseFormatConverter, ConverterReturn, verify_label_map

_VOC_CLASSES_DETECTION = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

_VOC_CLASSES_SEGMENTATION = ('__background__',) + _VOC_CLASSES_DETECTION
_SEGMENTATION_COLORS = ((
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
    (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128)
))


def reverse_label_map(label_map):
    return {value: key for key, value in label_map.items()}


def prepare_detection_labels(dataset_meta, has_background=True):
    labels_shift = 1 if has_background else 0
    if dataset_meta:
        meta = read_json(dataset_meta)
        if 'label_map' in meta:
            meta['label_map'] = verify_label_map(meta['label_map'])
            return reverse_label_map(meta['label_map'])
        if 'labels' in meta:
            labels = meta['labels']
            num_classes = len(labels)
            reversed_label_map = dict(zip(labels, list(range(labels_shift, num_classes + labels_shift))))
            if has_background:
                reversed_label_map['__background__'] = 0
            return reversed_label_map

    num_classes = len(_VOC_CLASSES_DETECTION)
    reversed_label_map = dict(zip(_VOC_CLASSES_DETECTION, list(range(labels_shift, num_classes + labels_shift))))
    if has_background:
        reversed_label_map['__background__'] = 0

    return reversed_label_map


class PascalVOCSegmentationConverter(BaseFormatConverter):
    __provider__ = 'voc_segmentation'
    annotation_types = (SegmentationAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'imageset_file': PathField(description="Path to file with validation image list."),
            'images_dir': PathField(
                optional=True, is_directory=True,
                description="Path to directory with images related to devkit root (default JPEGImages)."
            ),
            'mask_dir': PathField(
                optional=True, is_directory=True,
                description="Path to directory with ground truth segmentation masks related to devkit root "
                            "(default SegmentationClass)."
            ),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding)', optional=True
            )
        })

        return configuration_parameters

    def configure(self):
        self.image_set_file = self.get_value_from_config('imageset_file')
        self.image_dir = self.get_value_from_config('images_dir')
        dataset_meta_file = self.get_value_from_config('dataset_meta_file')
        self.dataset_meta = {} if not dataset_meta_file else read_json(dataset_meta_file)
        if not self.image_dir:
            self.image_dir = get_path(self.image_set_file.parents[-2] / 'JPEGImages', is_directory=True)

        self.mask_dir = self.config.get('mask_dir')
        if not self.mask_dir:
            self.mask_dir = get_path(self.image_set_file.parents[-2] / 'SegmentationClass', is_directory=True)

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_check_errors = [] if check_content else None
        annotations = []
        images_set = read_txt(self.image_set_file)
        num_iterations = len(images_set)
        for image_id, image in enumerate(images_set):
            image_file = '{}.jpg'.format(image)
            mask_file = '{}.png'.format(image)
            annotation = SegmentationAnnotation(image_file, mask_file, mask_loader=GTMaskLoader.SCIPY)
            annotations.append(annotation)
            if check_content:
                if not check_file_existence(self.image_dir / image_file):
                    content_check_errors.append('{}: does not exist'.format(self.image_dir / image_file))

                if not check_file_existence(self.mask_dir / mask_file):
                    content_check_errors.append('{}: does not exist'.format(self.image_dir / image_file))

            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id / num_iterations * 100)

        meta = {
            'label_map': self.dataset_meta.get('label_map', dict(enumerate(_VOC_CLASSES_SEGMENTATION))),
            'background_label': 0,
            'segmentation_colors': self.dataset_meta.get('segmentation_colors', _SEGMENTATION_COLORS)
        }

        return ConverterReturn(annotations, meta, content_check_errors)


class PascalVOCDetectionConverter(BaseFormatConverter):
    __provider__ = 'voc_detection'
    annotation_types = (DetectionAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'imageset_file': PathField(description="Path to file with validation image list."),
            'annotations_dir': PathField(is_directory=True, description="Path to directory with annotation files."),
            'images_dir': PathField(
                optional=True, is_directory=True,
                description="Path to directory with images related to devkit root (default JPEGImages)."
            ),
            'has_background': BoolField(
                optional=True, default=True, description="Allows convert dataset with/without adding background_label."
            ),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding)', optional=True
            )
        })
        return parameters

    def configure(self):
        self.image_set_file = self.get_value_from_config('imageset_file')
        self.image_dir = self.get_value_from_config('images_dir')
        if not self.image_dir:
            self.image_dir = get_path(self.image_set_file.parents[-2] / 'JPEGImages')
        self.annotations_dir = self.get_value_from_config('annotations_dir')
        self.has_background = self.get_value_from_config('has_background')
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        class_to_ind = prepare_detection_labels(self.dataset_meta, self.has_background)
        content_check_errors = [] if check_content else None

        detections = []
        image_set = read_txt(self.image_set_file, sep=None)
        num_iterations = len(image_set)
        for (image_id, image) in enumerate(image_set):
            root = read_xml(self.annotations_dir / '{}.xml'.format(image))

            identifier = root.find('.//filename').text
            get_path(self.image_dir / identifier)
            if check_content:
                if not check_file_existence(self.image_dir / identifier):
                    content_check_errors.append('{}: does not exist'.format(self.image_dir / identifier))

            labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []
            difficult_indices = []
            for entry in root:
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
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id / num_iterations * 100)

        meta = {'label_map': reverse_label_map(class_to_ind)}
        if self.has_background:
            meta['background_label'] = 0

        return ConverterReturn(detections, meta, content_check_errors)
