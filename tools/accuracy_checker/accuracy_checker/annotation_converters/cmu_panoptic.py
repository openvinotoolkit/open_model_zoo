"""
Copyright (c) 2018-2020 Intel Corporation

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

import cv2
import numpy as np

from ..utils import read_json, check_file_existence
from ..representation import PoseEstimation3dAnnotation
from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn


class CmuPanopticKeypointsConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'cmu_panoptic_keypoints'
    annotation_types = (PoseEstimation3dAnnotation,)

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        images_dir_name = 'hdImgs'
        labels_dir_name = 'hdPose3d_stage1_coco19'
        label_prefix = 'body3DScene'
        data, num_iterations = self._collect_data(self.data_dir, images_dir_name)

        keypoints_annotations = []
        content_errors = []
        image_id = 0
        for scene_dir, cameras_dir in data.items():
            calibration_name = 'calibration_{}.json'.format(scene_dir.stem)
            calibration = read_json(scene_dir / calibration_name)
            for camera_dir, images_paths in cameras_dir.items():
                camera_parameters = {}
                for camera in calibration['cameras']:
                    if camera['type'] != 'hd':
                        continue
                    if camera['name'] != camera_dir.stem:
                        continue
                    camera_parameters = {
                        'K': np.array(camera['K'], dtype=np.float32),
                        'distCoef': np.array(camera['distCoef'], dtype=np.float32),
                        'R': np.array(camera['R'], dtype=np.float32),
                        't': np.array(camera['t'], dtype=np.float32)}
                    break

                for image_path in images_paths:
                    if check_content:
                        if not check_file_existence(image_path):
                            content_errors.append('{}: does not exist'.format(image_path))
                    label_name = image_path.name.replace(camera_dir.stem, label_prefix)
                    label_path = scene_dir / labels_dir_name / label_name.replace('jpg', 'json')
                    label = read_json(label_path)

                    annotations = []
                    for body_id in range(len(label['bodies'])):
                        body = label['bodies'][body_id]
                        skel = np.array(body['joints19']).reshape((-1, 4)).transpose()
                        annotations.append({'id': body_id, 'body': skel})

                    keypoints_2d = CmuPanopticKeypointsConverter._project_3d_keypoints_to_frame(
                        annotations, camera_parameters)
                    keypoints_3d = CmuPanopticKeypointsConverter._to_camera_space(
                        annotations, camera_parameters['R'], camera_parameters['t'])

                    identifier = Path(*image_path.parts[-4:])
                    keypoints_annotation = PoseEstimation3dAnnotation(
                        identifier, keypoints_2d[:, 0].astype(np.float32), keypoints_2d[:, 1].astype(np.float32),
                        np.full_like(keypoints_2d[:, 1].astype(np.float32), 1),
                        x_3d_values=keypoints_3d[:, 0], y_3d_values=keypoints_3d[:, 1], z_3d_values=keypoints_3d[:, 2],
                        fx=camera_parameters['K'][0, 0])
                    keypoints_annotations.append(keypoints_annotation)
                    if progress_callback is not None and image_id & progress_interval == 0:
                        progress_callback(image_id / num_iterations * 100)
                    image_id += 1

        return ConverterReturn(keypoints_annotations, None, content_errors)

    @staticmethod
    def _project_3d_keypoints_to_frame(annotations, camera_parameters):
        keypoints_2d = []
        for annotation in annotations:
            pt = cv2.projectPoints(annotation['body'][0:3, :].transpose().copy(),
                                   cv2.Rodrigues(camera_parameters['R'])[0],
                                   camera_parameters['t'],
                                   camera_parameters['K'],
                                   camera_parameters['distCoef'])
            pt = np.squeeze(pt[0], axis=1).transpose()

            keypoints_2d.append(pt)

        return np.array(keypoints_2d)

    @staticmethod
    def _to_camera_space(annotations, R, t):
        keypoints_3d = np.zeros((len(annotations),
                                 annotations[0]['body'].shape[0],
                                 annotations[0]['body'].shape[1]), dtype=np.float32)
        for pose_id, annotation in enumerate(annotations):
            keypoints_in_camera_space = annotation['body']
            keypoints_in_camera_space[:3, :] = np.dot(R, keypoints_in_camera_space[:3, :]) + t
            keypoints_3d[pose_id] = keypoints_in_camera_space

        return keypoints_3d

    @staticmethod
    def _collect_data(data_dir, images_dir_name):
        data = {}
        num_iterations = 0
        for scene_dir in data_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            data[scene_dir] = {}
            scene_images_dir = scene_dir / images_dir_name
            for camera_dir in scene_images_dir.iterdir():
                if not camera_dir.is_dir():
                    continue
                data[scene_dir][camera_dir] = list(camera_dir.rglob('*.jpg'))
                num_iterations += len(data[scene_dir][camera_dir])

        return data, num_iterations
