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
import warnings

from ..config import PathField, BoolField
from ..representation import DetectionAnnotation, SegmentationAnnotation
from ..utils import read_xml, check_file_existence, read_json
from .format_converter import BaseFormatConverter, ConverterReturn


class LabelMeDetectionConverter(BaseFormatConverter):
    __provider__ = 'label_me_detection'
    annotation_types = (DetectionAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotations_dir': PathField(is_directory=True, description='Path to directory with annotation files.'),
            'images_dir': PathField(optional=True, is_directory=True, description='Path to directory with images.'),
            'dataset_meta_file': PathField(
                description='Path to json file with dataset meta (e.g. label_map, segmentation_colors)'
            ),
            'has_background': BoolField(
                optional=True, default=False, description='Allows convert dataset with/without adding background_label.'
            ),
        })
        return parameters

    def configure(self):
        self.image_dir = self.get_value_from_config('images_dir')
        self.annotations_dir = self.get_value_from_config('annotations_dir')
        meta_file = self.get_value_from_config('dataset_meta_file')
        self.dataset_meta = {} if not meta_file else read_json(meta_file)
        self.has_background = self.get_value_from_config('has_background')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_check_errors = [] if check_content else None

        detections = []
        annotation_files = list(self.annotations_dir.glob('*.xml'))
        num_iterations = len(annotation_files)

        label_map = get_label_map(self.dataset_meta.get('label_map'), self.has_background)
        reversed_label_map = reverse_label_map(label_map)

        for (idx, annotation_file) in enumerate(annotation_files):
            annotation = read_xml(annotation_file)

            identifier = annotation.find('.//filename').text

            labels, x_mins, y_mins, x_maxs, y_maxs = [], [], [], [], []

            for entry in annotation:
                if not entry.tag.startswith('object'):
                    continue

                label = reversed_label_map.get(entry.find('name').text)
                if label is None:
                    continue

                polygon = entry.find('polygon')
                if not polygon:
                    continue
                x_coords, y_coords = [], []
                points = polygon.findall('pt')
                if len(points) < 3:
                    warnings.warn('Object with less than 3 points is not a polygon')
                    continue
                for point in points:
                    x_coords.append(float(point.find('x').text))
                    y_coords.append(float(point.find('y').text))
                if len(x_coords) != len(y_coords):
                    raise ValueError('Number of polygon coordinates x and y are not equal')

                x_mins.append(min(x_coords))
                x_maxs.append(max(x_coords))
                y_mins.append(min(y_coords))
                y_maxs.append(max(y_coords))
                labels.append(label)

            if check_content:
                if not check_file_existence(self.image_dir / identifier):
                    content_check_errors.append('{}: does not exist'.format(self.image_dir / identifier))

            image_annotation = DetectionAnnotation(identifier, labels, x_mins, y_mins, x_maxs, y_maxs)

            detections.append(image_annotation)
            if progress_callback is not None and idx % progress_interval == 0:
                progress_callback(idx / num_iterations * 100)

        meta = {'label_map': label_map}

        if self.has_background:
            meta.update({
                'background_label': 0
            })

        return ConverterReturn(detections, meta, content_check_errors)


class LabelMeSegmentationConverter(BaseFormatConverter):
    __provider__ = 'label_me_segmentation'
    annotation_types = (SegmentationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotations_dir': PathField(is_directory=True, description='Path to directory with annotation files.'),
            'images_dir': PathField(optional=True, is_directory=True, description='Path to directory with images.'),
            'masks_dir': PathField(optional=True, is_directory=True,
                                   description='Path to directory with ground truth segmentation masks.'),
            'dataset_meta_file': PathField(
                description='Path to json file with dataset meta (e.g. label_map, segmentation_colors)'
            )
        })

        return parameters

    def configure(self):
        self.image_dir = self.get_value_from_config('images_dir')
        self.mask_dir = self.get_value_from_config('masks_dir')
        self.annotations_dir = self.get_value_from_config('annotations_dir')
        meta_file = self.get_value_from_config('dataset_meta_file')
        self.dataset_meta = {} if not meta_file else read_json(meta_file)

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        content_check_errors = [] if check_content else None

        annotations = []
        annotation_files = list(self.annotations_dir.glob('*.xml'))
        num_iterations = len(annotation_files)

        label_map = get_label_map(self.dataset_meta.get('label_map'))
        segmentation_colors = self.dataset_meta.get('segmentation_colors')
        if segmentation_colors is None:
            raise ValueError('segmentation_colors must be provided in dataset_meta_file')

        for (idx, annotation_file) in enumerate(annotation_files):
            annotation = read_xml(annotation_file)

            identifier = annotation.find('.//filename').text
            mask_file = None

            for entry in annotation:
                if not entry.tag.startswith('object'):
                    continue

                segm_object = entry.find('segm')
                if not segm_object:
                    continue
                mask_file = segm_object.find('mask').text

            if not mask_file:
                continue

            if check_content:
                if not check_file_existence(self.image_dir / identifier):
                    content_check_errors.append('{}: does not exist'.format(self.image_dir / identifier))

                if not check_file_existence(self.mask_dir / mask_file):
                    content_check_errors.append('{}: does not exist'.format(self.mask_dir / mask_file))

            image_annotation = SegmentationAnnotation(identifier, mask_file)

            annotations.append(image_annotation)
            if progress_callback is not None and idx % progress_interval == 0:
                progress_callback(idx / num_iterations * 100)

        meta = {
            'label_map': label_map,
            'segmentation_colors': segmentation_colors
        }

        return ConverterReturn(annotations, meta, content_check_errors)


def get_label_map(labels, has_background=False):
    if labels is None:
        raise ValueError('label_map must be provided in dataset_meta_file')

    if isinstance(labels, dict):
        return labels

    labels_offset = int(has_background)
    label_map = {}
    for idx, label_name in enumerate(labels):
        label_map[idx + labels_offset] = label_name
    if has_background:
        label_map[0] = 'background'
    return label_map


def reverse_label_map(label_map):
    return {value: key for key, value in label_map.items()}
