"""
Copyright (c) 2019 Intel Corporation

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

import numpy as np

from ..config import BoolField, PathField
from ..logging import print_info
from ..utils import read_json, convert_bboxes_xywh_to_x1y1x2y2, check_file_existence
from ..representation import (
    DetectionAnnotation, PoseEstimationAnnotation, CoCoInstanceSegmentationAnnotation, ContainerAnnotation
)
from .format_converter import BaseFormatConverter, FileBasedAnnotationConverter, ConverterReturn, verify_label_map

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def get_image_annotation(image_id, annotations_):
    return list(filter(lambda x: x['image_id'] == image_id, annotations_))


def get_label_map(dataset_meta, full_annotation, use_full_label_map=False, has_background=False):
    if dataset_meta:
        meta = read_json(dataset_meta)
        label_map = meta.get('label_map')
        if not label_map:
            labels = meta.get('labels')
            label_offset = int(has_background)
            if labels:
                label_map = {i + label_offset: label for i, label in enumerate(labels)}
        if label_map:
            label_map = verify_label_map(label_map)
            label_id_to_label = {i: i for i in label_map}
            return label_map, label_id_to_label

    labels = full_annotation['categories']

    if not use_full_label_map:
        label_offset = 1 if has_background else 0
        label_id_to_label = {label['id']: label_id + label_offset for label_id, label in enumerate(labels)}
        label_map = {label_id + label_offset: label['name'] for label_id, label in enumerate(labels)}
    else:
        label_id_to_label = {label['id']: label['id'] for label in labels}
        label_map = {label['id']: label['name'] for label in labels}

    return label_map, label_id_to_label


class MSCocoDetectionConverter(BaseFormatConverter):
    __provider__ = 'mscoco_detection'
    annotation_types = (DetectionAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'annotation_file': PathField(description="Path to annotation file in json format."),
            'use_full_label_map': BoolField(
                optional=True, default=False,
                description="Allows to use original label map (with 91 object categories) "
                            "from paper instead public available (80 categories)."
            ),
            'has_background': BoolField(
                optional=True, default=False, description="Allows convert dataset with/without adding background_label."
            ),
            'sort_annotations': BoolField(
                optional=True, default=True, description='Allows to sort annotations before conversion'
            ),
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            ),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding)', optional=True
            )
        })
        return configuration_parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.has_background = self.get_value_from_config('has_background')
        self.use_full_label_map = self.get_value_from_config('use_full_label_map')
        self.sort_annotations = self.get_value_from_config('sort_annotations')
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        full_annotation = read_json(self.annotation_file)
        image_info = full_annotation['images']
        image_ids = [(image['id'], image['file_name'], np.array([image['height'], image['width'], 3]))
                     for image in image_info]
        if self.sort_annotations:
            image_ids.sort(key=lambda value: value[0])
        annotations = full_annotation['annotations']

        label_map, label_id_to_label = get_label_map(
            self.dataset_meta, full_annotation, self.use_full_label_map, self.has_background
        )

        meta = {}
        if self.has_background:
            label_map[0] = 'background'
            meta['background_label'] = 0

        meta.update({'label_map': label_map})
        detection_annotations, content_errors = self._create_representations(
            image_ids, annotations, label_id_to_label, check_content, progress_callback, progress_interval
        )

        return ConverterReturn(detection_annotations, meta, content_errors)

    def _create_representations(
            self, image_info, annotations, label_id_to_label, check_content, progress_callback, progress_interval
    ):
        detection_annotations = []
        content_errors = [] if check_content else None
        num_iterations = len(image_info)
        image_iter = tqdm(enumerate(image_info)) if tqdm is not None else enumerate(image_info)

        for (image_id, image) in image_iter:
            image_labels, xmins, ymins, xmaxs, ymaxs, is_crowd, _ = self._read_image_annotation(
                image, annotations,
                label_id_to_label
            )
            if check_content:
                image_full_path = self.images_dir / image[1]
                if not check_file_existence(image_full_path):
                    content_errors.append('{}: does not exist'.format(image_full_path))
            detection_annotation = DetectionAnnotation(image[1], image_labels, xmins, ymins, xmaxs, ymaxs)
            detection_annotation.metadata['iscrowd'] = is_crowd
            detection_annotations.append(detection_annotation)
            if tqdm is None and image_id % progress_interval == 0:
                print_info('{} / {} processed'.format(image_id, num_iterations))
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id / num_iterations * 100)

        return detection_annotations, content_errors

    @staticmethod
    def _read_image_annotation(image, annotations, label_id_to_label):
        image_annotation = get_image_annotation(image[0], annotations)
        image_labels = [label_id_to_label[annotation['category_id']] for annotation in image_annotation]
        xmins = [annotation['bbox'][0] for annotation in image_annotation]
        ymins = [annotation['bbox'][1] for annotation in image_annotation]
        widths = [annotation['bbox'][2] for annotation in image_annotation]
        heights = [annotation['bbox'][3] for annotation in image_annotation]
        xmaxs = np.add(xmins, widths)
        ymaxs = np.add(ymins, heights)
        is_crowd = [annotation['iscrowd'] for annotation in image_annotation]
        segmentation_polygons = [annotation['segmentation'] for annotation in image_annotation]

        return image_labels, xmins, ymins, xmaxs, ymaxs, is_crowd, segmentation_polygons


class MSCocoKeypointsConverter(FileBasedAnnotationConverter):
    __provider__ = 'mscoco_keypoints'
    annotation_types = (PoseEstimationAnnotation, )
    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'images_dir': PathField(
                    is_directory=True, optional=True,
                    description='path to dataset images, used only for content existence check'
                ),
                'dataset_meta_file': PathField(
                    description='path to json file with dataset meta (e.g. label_map, color_encoding)', optional=True
                )
            }
        )
        return parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        keypoints_annotations = []
        content_errors = []

        full_annotation = read_json(self.annotation_file)
        image_info = full_annotation['images']
        annotations = full_annotation['annotations']
        label_map, _ = get_label_map(self.dataset_meta, full_annotation, True)
        num_iterations = len(image_info)
        for image_id, image in enumerate(image_info):
            identifier = image['file_name']
            if check_content:
                full_image_path = self.images_dir / identifier
                if not check_file_existence(full_image_path):
                    content_errors.append('{}: does not exist'.format(full_image_path))
            image_annotation = get_image_annotation(image['id'], annotations)
            if not image_annotation:
                continue
            x_vals, y_vals, visibility, labels, areas, is_crowd, bboxes, difficult = [], [], [], [], [], [], [], []
            for target in image_annotation:
                if target['num_keypoints'] == 0:
                    difficult.append(len(x_vals))
                labels.append(target['category_id'])
                keypoints = target['keypoints']
                x_vals.append(keypoints[::3])
                y_vals.append(keypoints[1::3])
                visibility.append(keypoints[2::3])
                areas.append(target['area'])
                bboxes.append(convert_bboxes_xywh_to_x1y1x2y2(*target['bbox']))
                is_crowd.append(target['iscrowd'])
            keypoints_annotation = PoseEstimationAnnotation(
                identifier, np.array(x_vals), np.array(y_vals), np.array(visibility), np.array(labels)
            )
            keypoints_annotation.metadata['areas'] = areas
            keypoints_annotation.metadata['rects'] = bboxes
            keypoints_annotation.metadata['iscrowd'] = is_crowd
            keypoints_annotation.metadata['difficult_boxes'] = difficult

            keypoints_annotations.append(keypoints_annotation)
            if progress_callback is not None and image_id & progress_interval == 0:
                progress_callback(image_id / num_iterations * 100)

        return ConverterReturn(keypoints_annotations, {'label_map': label_map}, None)


class MSCocoSegmentationConverter(MSCocoDetectionConverter):
    __provider__ = 'mscoco_segmentation'
    annotation_types = (CoCoInstanceSegmentationAnnotation, )

    def _create_representations(
            self, image_info, annotations, label_id_to_label, check_content, progress_callback, progress_interval
    ):
        segmentation_annotations = []
        content_errors = None if not check_content else []
        num_iterations = len(image_info)
        image_iter = tqdm(enumerate(image_info)) if tqdm is not None else enumerate(image_info)

        for (image_id, image) in image_iter:
            image_labels, _, _, _, _, is_crowd, segmentations = self._read_image_annotation(
                image, annotations,
                label_id_to_label
            )
            annotation = CoCoInstanceSegmentationAnnotation(image[1], segmentations, image_labels)
            if check_content:
                image_full_path = self.images_dir / image[1]
                if not check_file_existence(image_full_path):
                    content_errors.append('{}: does not exist'.format(image_full_path))
            annotation.metadata['iscrowd'] = is_crowd
            annotation.metadata['image_size'] = image[2]
            segmentation_annotations.append(annotation)
            if tqdm is None and image_id % progress_interval == 0:
                print_info('{} / {} processed'.format(image_id, num_iterations))

            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id / num_iterations * 100)

        return segmentation_annotations, content_errors


class MSCocoMaskRCNNConverter(MSCocoDetectionConverter):
    __provider__ = 'mscoco_mask_rcnn'
    annotation_types = (DetectionAnnotation, CoCoInstanceSegmentationAnnotation)

    def _create_representations(
            self, image_info, annotations, label_id_to_label, check_content, progress_callback, progress_interval
    ):
        container_annotations = []
        content_errors = None if not check_content else []
        num_iterations = len(image_info)
        image_iter = tqdm(enumerate(image_info)) if tqdm is not None else enumerate(image_info)

        for (image_id, image) in image_iter:
            image_labels, xmins, ymins, xmaxs, ymaxs, is_crowd, segmentations = self._read_image_annotation(
                image, annotations,
                label_id_to_label
            )
            if check_content:
                image_full_path = self.images_dir / image[1]
                if not check_file_existence(image_full_path):
                    content_errors.append('{}: does not exist'.format(image_full_path))
            detection_annotation = DetectionAnnotation(image[1], image_labels, xmins, ymins, xmaxs, ymaxs)
            detection_annotation.metadata['iscrowd'] = is_crowd
            segmentation_annotation = CoCoInstanceSegmentationAnnotation(image[1], segmentations, image_labels)
            segmentation_annotation.metadata['iscrowd'] = is_crowd
            segmentation_annotation.metadata['rects'] = np.c_[xmins, ymins, xmaxs, ymaxs]
            segmentation_annotation.metadata['image_size'] = image[2]
            container_annotations.append(ContainerAnnotation({
                'detection_annotation': detection_annotation,
                'segmentation_annotation': segmentation_annotation
            }))

            if tqdm is None and image_id % progress_interval == 0:
                print_info('{} / {} processed'.format(image_id, num_iterations))

            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id / num_iterations * 100)

        return container_annotations, content_errors


class MSCocoSingleKeypointsConverter(FileBasedAnnotationConverter):
    __provider__ = 'mscoco_single_keypoints'
    annotation_types = (PoseEstimationAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'images_dir': PathField(
                    is_directory=True, optional=True,
                    description='path to dataset images, used only for content existence check'
                )
            }
        )
        return parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        keypoints_annotations = []
        content_errors = []

        full_annotation = read_json(self.annotation_file)
        image_info = full_annotation['images']
        annotations = full_annotation['annotations']
        num_iterations = len(image_info)
        for image_id, image in enumerate(image_info):
            identifier = image['file_name']
            if check_content:
                full_image_path = self.images_dir / identifier
                if not check_file_existence(full_image_path):
                    content_errors.append('{}: does not exist'.format(full_image_path))
            image_annotation = get_image_annotation(image['id'], annotations)
            if not image_annotation:
                continue
            for target in image_annotation:
                x_vals, y_vals, visibility, labels, areas, is_crowd, bboxes, difficult = [], [], [], [], [], [], [], []
                if target['num_keypoints'] == 0:
                    continue
                labels.append(target['category_id'])
                keypoints = target['keypoints']
                x_vals.append(keypoints[::3])
                y_vals.append(keypoints[1::3])
                visibility.append(keypoints[2::3])
                areas.append(target['area'])
                bboxes.append(target['bbox'])
                is_crowd.append(target['iscrowd'])
                keypoints_annotation = PoseEstimationAnnotation(
                    identifier, np.array(x_vals), np.array(y_vals), np.array(visibility), np.array(labels)
                )
                keypoints_annotation.metadata['areas'] = areas
                keypoints_annotation.metadata['rects'] = bboxes
                keypoints_annotation.metadata['iscrowd'] = is_crowd
                keypoints_annotation.metadata['difficult_boxes'] = difficult

                keypoints_annotations.append(keypoints_annotation)
                if progress_callback is not None and image_id & progress_interval == 0:
                    progress_callback(image_id / num_iterations * 100)

        return ConverterReturn(keypoints_annotations, {'label_map': {1: 'person'}}, content_errors)
