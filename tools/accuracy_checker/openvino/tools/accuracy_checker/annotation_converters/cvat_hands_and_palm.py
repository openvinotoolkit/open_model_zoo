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
from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import DetectionAnnotation, HandLandmarksAnnotation
from ..utils import check_file_existence, read_json
from ..config import PathField


class CVATHandLandmarkConverter(FileBasedAnnotationConverter):
    __provider__ = 'cvat_hand_landmark'

    annotation_types = (DetectionAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            ),
            'bbox_file': PathField(
                is_directory=False, optional=False,
                description='path to file with palm bounding box data'
            ),
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.bbox_file = self.get_value_from_config('bbox_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = read_json(self.annotation_file)
        bboxes = read_json(self.bbox_file)
        # annotation_meta = annotation.find('meta')
        # size = int(annotation_meta.find('task').find('size').text)
        # label_to_id, meta = self.generate_labels_mapping(annotation_meta)
        label_to_id, meta = self.generate_labels_mapping(annotation['categories'])
        num_landmarks = len(meta['label']) - 1
        annotations = []
        content_errors = None if not check_content else []
        for image_id, image in enumerate(annotation['images']):
            keypoints = [t for t in annotation['annotations'] if t['image_id'] == image['id']]
            assert len(keypoints) == 21
            bb_images = [t for t in annotation['images'] if t['file_name'] == image['file_name']]
            assert len(bb_images) == 1
            bbs = [t for t in bboxes['annotations'] if t['image_id'] == bb_images[0]['id']]
            assert len(bbs) == 1
            bbox = bbs[0]['bbox']

            identifier = image['file_name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            landmarks_x, landmarks_y = self.get_landmarks(keypoints, num_landmarks)
            landmarks_annotation = HandLandmarksAnnotation(identifier, np.array(landmarks_x), np.array(landmarks_y))
            landmarks_annotation.metadata['rect'] = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            annotations.append(landmarks_annotation)
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id * 100 / len(annotation['images']))

        return ConverterReturn(annotations, meta, content_errors)

    @staticmethod
    def get_landmarks(image_annotation, num_landmarks):
        landmarks_x, landmarks_y = np.zeros(num_landmarks), np.zeros(num_landmarks)
        for point in image_annotation:
            idx = int(point.get('category_id')) - 1
            keypoints = point.get('keypoints')
            assert len(keypoints) >= 2
            landmarks_x[idx] = float(keypoints[0])
            landmarks_y[idx] = float(keypoints[1])

        return landmarks_x, landmarks_y

    @staticmethod
    def generate_labels_mapping(categories):
        label_to_id = {t['name']: t['id'] for t in categories}
        meta = {'label_map': {value: key for key, value in label_to_id.items()}, 'label': list(label_to_id.keys())}
        # if self.dataset_meta:
        #     meta = read_json(self.dataset_meta)
        #     if 'labels' in meta and 'label_map' not in meta:
        #         offset = int(self.has_background)
        #         label_to_id = {label_name: label_id + offset for label_id, label_name in enumerate(meta['labels'])}
        #         meta['label_map'] = {'label_map': {value: key for key, value in label_to_id.items()}}
        #         if self.has_background:
        #             meta['label_map'][0] = 'background'
        #             meta['background_label'] = 0
        #
        #     label_map = meta.get('label_map')
        #     if not label_map:
        #         raise ConfigError('dataset_meta_file should contains labels or label_map')
        #     label_to_id = {value: key for key, value in label_map.items()}
        #
        #     return label_to_id, meta
        #
        # meta = {}
        # if self.label_map_file:
        #     label_to_id = read_json(self.label_map_file).get('labels')
        #     if not label_to_id:
        #         raise ConfigError('label_map_file does not contains labels key')
        # else:
        #     labels = [label.find('name').text for label in annotation_meta.iter('label') if label.find('name').text]
        #     if not labels:
        #         raise ConfigError('annotation file does not contains labels')
        #     if self.has_background:
        #         labels = ['background'] + labels
        #         meta['background_label'] = 0
        #     label_to_id = {label: idx for idx, label in enumerate(labels)}
        # meta['label_map'] = {value: key for key, value in label_to_id.items()}

        return label_to_id, meta


class CVATPalmDetectionConverter(FileBasedAnnotationConverter):
    __provider__ = 'cvat_palm_detection'
    annotation_types = (DetectionAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            ),
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = read_json(self.annotation_file)
        # annotation_meta = annotation.find('meta')
        # size = int(annotation_meta.find('task').find('size').text)
        # label_to_id, meta = self.generate_labels_mapping(annotation_meta)
        annotations = []
        content_errors = None if not check_content else []
        for ann_id, ann in enumerate(annotation['annotations']):
            images = [t for t in annotation['images'] if t['id'] == ann['image_id']]
            assert len(images) == 1
            image = images[0]
            identifier = image['file_name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            bbox = ann['bbox']
            detection_annotation = DetectionAnnotation(identifier, None, bbox[0], bbox[1], bbox[0] + bbox[2],
                                                       bbox[1] + bbox[3])
            annotations.append(detection_annotation)
            if progress_callback is not None and ann_id % progress_interval == 0:
                progress_callback(ann_id * 100 / len(annotation['annotations']))

        return ConverterReturn(annotations, None, content_errors)
