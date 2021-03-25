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

import numpy as np
from PIL import Image

from ..config import BoolField, PathField, StringField
from ..utils import read_json, convert_bboxes_xywh_to_x1y1x2y2, check_file_existence
from ..representation import (
    DetectionAnnotation,
    PoseEstimationAnnotation,
    CoCoInstanceSegmentationAnnotation,
    ContainerAnnotation,
    SegmentationAnnotation,
)
from .format_converter import BaseFormatConverter, FileBasedAnnotationConverter, ConverterReturn, verify_label_map
from ..progress_reporters import PrintProgressReporter

from ..utils import UnsupportedPackage

try:
    import pycocotools.mask as maskUtils
except ImportError as import_error:
    maskUtils = UnsupportedPackage("pycocotools", import_error.msg)

from ..representation.segmentation_representation import GTMaskLoader
from  ..data_readers import MultiInstanceIdentifier

from ..logging import warning


COCO_TO_VOC = {
    1: 15,  # person
    2: 2,  # bicycle
    3: 7,  # car
    4: 14,  # motorbike
    5: 1,  # airplane
    6: 6,  # bus
    7: 19,  # train
    9: 4,  # boat
    16: 3,  # bird
    17: 8,  # cat
    18: 12,  # dog
    19: 13,  # horse
    20: 17,  # sheep
    21: 10,  # cow
    44: 5,  # bottle
    62: 9,  # chair
    63: 18,  # couch/sofa
    64: 16,  # potted plant
    67: 11,  # dining table
    72: 20,  # tv
}


def get_image_annotation(image_id, annotations_):
    return list(filter(lambda x: x['image_id'] == image_id, annotations_))


def get_label_map(dataset_meta, full_annotation, use_full_label_map=False, has_background=False,
                  convert_COCO_to_VOC_labels=False):
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
            label_id_to_label = {i: i for i in label_map} if not convert_COCO_to_VOC_labels else {
                i: COCO_TO_VOC[i] for i in label_map}
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


sort_lambda = {
    'image_id': lambda value: value[0],
    'image_size': lambda value: tuple(value[2])
}


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
            'sort_key': StringField(
                optional=True, default='image_id', choices=['image_id', 'image_size'],
                description='Key by which annotations will be sorted.'
            ),
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            ),
            'dataset_meta_file': PathField(
                description='path to json file with dataset meta (e.g. label_map, color_encoding)', optional=True
            ),
            'convert_COCO_to_VOC_labels': BoolField(
                optional=True, default=False,
                description="Allows to convert COCO labels to Pacsal VOC labels."
            ),
        })
        return configuration_parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.has_background = self.get_value_from_config('has_background')
        self.use_full_label_map = self.get_value_from_config('use_full_label_map')
        self.sort_annotations = self.get_value_from_config('sort_annotations')
        self.sort_key = self.get_value_from_config('sort_key')
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')
        self.convert_COCO_to_VOC_labels = self.get_value_from_config('convert_COCO_to_VOC_labels')
        self.meta = {}

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        full_annotation = read_json(self.annotation_file)
        image_info = full_annotation['images']
        image_ids = [(image['id'], image['file_name'], np.array([image['height'], image['width'], 3]))
                     for image in image_info]
        if self.sort_annotations:
            image_ids.sort(key=sort_lambda[self.sort_key])

        annotations = full_annotation['annotations']

        label_map, label_id_to_label = get_label_map(
            self.dataset_meta, full_annotation, self.use_full_label_map, self.has_background,
            self.convert_COCO_to_VOC_labels
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
        progress_reporter = PrintProgressReporter(print_interval=progress_interval)
        progress_reporter.reset(num_iterations, 'annotations')

        for (image_id, image) in enumerate(image_info):
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
            progress_reporter.update(image_id, 1)

        progress_reporter.finish()

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
                ),
                'sort_annotations': BoolField(
                    optional=True, default=True, description='Allows to sort annotations before conversion'
                ),
                'sort_key': StringField(
                    optional=True, default='image_id', choices=['image_id', 'image_size'],
                    description='Key by which annotations will be sorted.'
                ),
                'remove_empty_images': BoolField(
                    optional=True, default=False,
                    description='Allows excluding/inclusing images without objects from/to the dataset.'
                )
            }
        )
        return parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')
        self.sort_annotations = self.get_value_from_config('sort_annotations')
        self.sort_key = self.get_value_from_config('sort_key')
        self.remove_empty_images = self.get_value_from_config('remove_empty_images')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        keypoints_annotations = []
        content_errors = []

        full_annotation = read_json(self.annotation_file)
        image_info = full_annotation['images']

        image_ids = [(image['id'], image['file_name'], np.array([image['height'], image['width'], 3]))
                     for image in image_info]
        if self.sort_annotations:
            image_ids.sort(key=sort_lambda[self.sort_key])

        annotations = full_annotation['annotations']
        label_map, _ = get_label_map(self.dataset_meta, full_annotation, True)
        num_iterations = len(image_info)
        for image_id, image in enumerate(image_ids):
            identifier = image[1]
            if check_content:
                full_image_path = self.images_dir / identifier
                if not check_file_existence(full_image_path):
                    content_errors.append('{}: does not exist'.format(full_image_path))
            image_annotation = get_image_annotation(image[0], annotations)
            if not image_annotation and self.remove_empty_images:
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

def make_segmentation_mask(height, width, path_to_mask, labels, segmentations):
    polygons = []
    for mask in segmentations:
        if isinstance(mask, list):
            rles = maskUtils.frPyObjects(mask, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(mask['counts'], list):
            rle = maskUtils.frPyObjects(mask, height, width)
        else:
            rle = mask
        polygons.append(rle)

    masks = []
    for polygon in polygons:
        mask = maskUtils.decode(polygon)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=-1)
    else:
        masks = np.zeros((height, width, 0), dtype=np.uint8)
    masks = (masks * np.asarray(labels, dtype=np.uint8)).max(axis=-1)
    mask = np.squeeze(masks)

    image = Image.frombytes('L', (width, height), mask.tostring())
    image.save(path_to_mask)

    return mask


class MSCocoSegmentationConverter(MSCocoDetectionConverter):
    __provider__ = 'mscoco_segmentation'
    annotation_types = (CoCoInstanceSegmentationAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'semantic_only': BoolField(
                optional=True, default=False, description="Semantic segmentation only mode."
            ),
            'masks_dir': PathField(
                is_directory=True, optional=True,
                check_exists=False,
                default='./segmentation_masks',
                description='path to segmentation masks, used if semantic_only is True'
            ),
            'convert_mask': BoolField(
                optional=True,
                default=True,
                description="Allows to convert segmentation mask."
            ),
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.semantic_only = self.get_value_from_config('semantic_only')
        self.masks_dir = self.get_value_from_config('masks_dir')
        self.convert_mask = self.get_value_from_config('convert_mask')

    def _create_representations(
            self, image_info, annotations, label_id_to_label, check_content, progress_callback, progress_interval
    ):
        segmentation_annotations = []
        content_errors = None if not check_content else []
        num_iterations = len(image_info)
        progress_reporter = PrintProgressReporter(print_interval=progress_interval)
        progress_reporter.reset(num_iterations, 'annotations')
        if self.semantic_only and self.convert_mask:
            if not self.masks_dir.exists():
                self.masks_dir.mkdir()
                warning('Segmentation masks will be located in {} folder'.format(str(self.masks_dir.resolve())))

        for (image_id, image) in enumerate(image_info):
            image_labels, is_crowd, segmentations = self._read_image_annotation(
                image, annotations,
                label_id_to_label
            )

            if not image_labels:
                continue

            if not self.semantic_only:
                annotation = CoCoInstanceSegmentationAnnotation(image[1], segmentations, image_labels)
            else:
                h, w, _ = image[2]
                mask_file = self.masks_dir / "{:012}.png".format(image[0])
                if self.convert_mask:
                    make_segmentation_mask(h, w, mask_file, image_labels, segmentations)
                annotation = SegmentationAnnotation(image[1],
                                                    str(mask_file.relative_to(self.masks_dir.parent)),
                                                    mask_loader=GTMaskLoader.PILLOW)

            if check_content:
                image_full_path = self.images_dir / image[1]
                if not check_file_existence(image_full_path):
                    content_errors.append('{}: does not exist'.format(image_full_path))
            annotation.metadata['iscrowd'] = is_crowd
            annotation.metadata['image_size'] = image[2]
            segmentation_annotations.append(annotation)
            progress_reporter.update(image_id, 1)

        progress_reporter.finish()

        return segmentation_annotations, content_errors

    @staticmethod
    def _read_image_annotation(image, annotations, label_id_to_label):
        image_annotation = get_image_annotation(image[0], annotations)
        mask = [label_id_to_label.get(annotation['category_id']) is not None for annotation in image_annotation]
        image_labels = [label_id_to_label[annotation['category_id']]
                        for index, annotation in enumerate(image_annotation) if mask[index]]
        is_crowd = [annotation['iscrowd'] for index, annotation in enumerate(image_annotation) if mask[index]]
        segmentation_polygons = [annotation['segmentation']
                                 for index, annotation in enumerate(image_annotation) if mask[index]]

        return image_labels, is_crowd, segmentation_polygons


class MSCocoMaskRCNNConverter(MSCocoDetectionConverter):
    __provider__ = 'mscoco_mask_rcnn'
    annotation_types = (DetectionAnnotation, CoCoInstanceSegmentationAnnotation)

    def _create_representations(
            self, image_info, annotations, label_id_to_label, check_content, progress_callback, progress_interval
    ):
        container_annotations = []
        content_errors = None if not check_content else []
        num_iterations = len(image_info)
        progress_reporter = PrintProgressReporter(print_interval=progress_interval)
        progress_reporter.reset(num_iterations, 'annotations')

        for (image_id, image) in enumerate(image_info):
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
            progress_reporter.update(image_id, 1)

        progress_reporter.finish()

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
                ),
                'sort_annotations': BoolField(
                    optional=True, default=True, description='Allows to sort annotations before conversion'
                ),
                'sort_key': StringField(
                    optional=True, default='image_id', choices=['image_id', 'image_size'],
                    description='Key by which annotations will be sorted.'
                )
            }
        )
        return parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.sort_annotations = self.get_value_from_config('sort_annotations')
        self.sort_key = self.get_value_from_config('sort_key')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        keypoints_annotations = []
        content_errors = []

        full_annotation = read_json(self.annotation_file)
        image_info = full_annotation['images']

        image_ids = [(image['id'], image['file_name'], np.array([image['height'], image['width'], 3]))
                     for image in image_info]
        if self.sort_annotations:
            image_ids.sort(key=sort_lambda[self.sort_key])

        annotations = full_annotation['annotations']
        num_iterations = len(image_info)
        for image_id, image in enumerate(image_ids):
            identifier = image[1]
            if check_content:
                full_image_path = self.images_dir / identifier
                if not check_file_existence(full_image_path):
                    content_errors.append('{}: does not exist'.format(full_image_path))
            image_annotation = get_image_annotation(image[0], annotations)
            if not image_annotation:
                continue
            for target_id, target in enumerate(image_annotation):
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
                idx = MultiInstanceIdentifier(identifier, target_id)
                keypoints_annotation = PoseEstimationAnnotation(
                    idx, np.array(x_vals), np.array(y_vals), np.array(visibility), np.array(labels)
                )
                keypoints_annotation.metadata['areas'] = areas
                keypoints_annotation.metadata['rects'] = bboxes
                keypoints_annotation.metadata['iscrowd'] = is_crowd
                keypoints_annotation.metadata['difficult_boxes'] = difficult

                keypoints_annotations.append(keypoints_annotation)
                if progress_callback is not None and image_id & progress_interval == 0:
                    progress_callback(image_id / num_iterations * 100)

        return ConverterReturn(keypoints_annotations, {'label_map': {1: 'person'}}, content_errors)
