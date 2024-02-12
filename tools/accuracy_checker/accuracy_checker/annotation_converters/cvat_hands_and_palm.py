"""
Copyright (c) 2018-2024 Intel Corporation

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
from ..config import PathField, BoolField, NumberField


class CVATHandPalmConverterBase:
    @staticmethod
    def get_landmarks(image_annotation, num_landmarks):
        landmarks_x, landmarks_y = np.zeros(num_landmarks), np.zeros(num_landmarks)
        for point in image_annotation:
            idx = int(point.get('category_id')) - 1
            keypoints = point.get('keypoints')
            if len(keypoints) < 2:
                raise ValueError("Invalid annotation")
            landmarks_x[idx] = float(keypoints[0])
            landmarks_y[idx] = float(keypoints[1])

        return np.array(landmarks_x), np.array(landmarks_y)


class CVATHandLandmarkConverter(CVATHandPalmConverterBase, FileBasedAnnotationConverter):
    __provider__ = 'cvat_hand_landmark'

    annotation_types = (HandLandmarksAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            'images_dir': PathField(is_directory=True, optional=True,
                                    description='path to dataset images, used only for content existence check'),
            'bbox_file': PathField(is_directory=False, optional=False,
                                   description='path to file with palm bounding box data'),
            'from_landmarks': BoolField(optional=True, default=False,
                                        description='acquire bounding box data from landmarks'),
            'padding': NumberField(optional=True, default=10, value_type=int,
                                   description='additional padding while acquiring bounding box data from landmarks'),
            'num_keypoints': NumberField(optional=True, default=21, value_type=int,
                                   description='Number of keypoints in dataset annotation'),
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.bbox_file = self.get_value_from_config('bbox_file')
        self.from_landmarks = self.get_value_from_config('from_landmarks')
        self.padding = self.get_value_from_config('padding')
        self.num_keypoints = self.get_value_from_config('num_keypoints')
        self.annotation = read_json(self.annotation_file)

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        meta = self.get_meta()
        bboxes = read_json(self.bbox_file)
        num_landmarks = len(meta['label']) - 1
        annotations = []
        content_errors = None if not check_content else []
        for image_id, image in enumerate(self.annotation['images']):
            keypoints = [t for t in self.annotation['annotations'] if t['image_id'] == image['id']]
            if len(keypoints) != self.num_keypoints:
                if check_content:
                    content_errors.append('Invalid number of keypoints in annotation: {}', len(keypoints))
                continue
            bb_images = [t for t in self.annotation['images'] if t['file_name'] == image['file_name']]
            if len(bb_images) != 1:
                if check_content:
                    content_errors.append('Invalid number of annotations for image {}', image['file_name'])
                continue
            bbs = [t for t in bboxes['annotations'] if t['image_id'] == bb_images[0]['id']]
            if len(bbs) != 1:
                if check_content:
                    content_errors.append('Invalid number of bounding box in annotation for image {}',
                                          image['file_name'])
                continue
            bbox = bbs[0]['bbox']

            identifier = image['file_name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            landmarks_x, landmarks_y = self.get_landmarks(keypoints, num_landmarks)
            landmarks_annotation = HandLandmarksAnnotation(identifier, landmarks_x, landmarks_y)
            if self.from_landmarks:
                landmarks_annotation.metadata['rect'] = [np.min(landmarks_x) - self.padding,
                                                         np.min(landmarks_y) - self.padding,
                                                         np.max(landmarks_x) + self.padding,
                                                         np.max(landmarks_y) + self.padding]
            else:
                landmarks_annotation.metadata['rect'] = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            annotations.append(landmarks_annotation)
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id * 100 / len(self.annotation['images']))

        return ConverterReturn(annotations, meta, content_errors)

    def get_meta(self):
        categories = self.annotation['categories']
        label_to_id = {t['name']: t['id'] for t in categories}
        return {'label_map': {value: key for key, value in label_to_id.items()},
                'label': list(label_to_id.keys()),
                'wrist_id': [t['id'] for t in categories if t['name'] == 'WRIST'][0] - 1,
                'mf_mcp_id': [t['id'] for t in categories if t['name'] == 'MIDDLE_FINGER_MCP'][0] - 1,
                'label_to_id': label_to_id
                }


class CVATPalmDetectionConverter(FileBasedAnnotationConverter, CVATHandPalmConverterBase):
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
            'landmarks_file': PathField(
                is_directory=False, optional=True,
                description='path to file with hand landmarks data'
            ),
            'padding': NumberField(optional=True, default=40, value_type=int,
                                   description='additional padding while acquiring bounding box data from landmarks'
                                   ),
        })
        return configuration_parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent
        self.landmarks_file = self.get_value_from_config('landmarks_file')
        self.padding = self.get_value_from_config('padding')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        if self.landmarks_file is not None:
            return self.convert_from_landmarks(check_content, progress_callback, progress_interval, **kwargs)

        annotation = read_json(self.annotation_file)
        annotations = []
        content_errors = None if not check_content else []
        for ann_id, ann in enumerate(annotation['annotations']):
            images = [t for t in annotation['images'] if t['id'] == ann['image_id']]
            if len(images) != 1:
                raise ValueError('Invalid annotation for image: {}'.format(ann['image_id']))
            image = images[0]
            identifier = image['file_name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            bbox = ann['bbox']
            detection_annotation = DetectionAnnotation(identifier, [0],
                                                       [bbox[0]],
                                                       [bbox[1]],
                                                       [bbox[0] + bbox[2]],
                                                       [bbox[1] + bbox[3]])
            annotations.append(detection_annotation)
            if progress_callback is not None and ann_id % progress_interval == 0:
                progress_callback(ann_id * 100 / len(annotation['annotations']))

        return ConverterReturn(annotations, self.get_meta(), content_errors)

    def convert_from_landmarks(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        self.annotation = read_json(self.landmarks_file)
        meta = self.get_meta()
        num_landmarks = len(meta['label']) - 1
        annotations = []
        content_errors = None if not check_content else []
        for image_id, image in enumerate(self.annotation['images']):
            keypoints = [t for t in self.annotation['annotations'] if t['image_id'] == image['id']]
            if len(keypoints) != num_landmarks:
                raise ValueError('Invalid annotation for image {}'.format(image['id']))

            identifier = image['file_name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            landmarks_x, landmarks_y = self.get_landmarks(keypoints, num_landmarks)
            detection_annotation = DetectionAnnotation(identifier, [0, ],
                                                       [np.min(landmarks_x) - self.padding],
                                                       [np.min(landmarks_y) - self.padding],
                                                       [np.max(landmarks_x) + self.padding],
                                                       [np.max(landmarks_y) + self.padding])
            annotations.append(detection_annotation)
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id * 100 / len(self.annotation['images']))

        return ConverterReturn(annotations, self.get_meta(), content_errors)

    def get_meta(self):
        return {'label_map': {0: 'PALM'}, 'label': ['PALM']}
