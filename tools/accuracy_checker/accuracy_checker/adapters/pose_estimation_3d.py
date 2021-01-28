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

from collections import OrderedDict

import cv2
import numpy as np

from ..adapters import Adapter
from ..adapters.pose_estimation import HumanPoseAdapter
from ..config import ConfigValidator, StringField
from ..representation import PoseEstimationPrediction, PoseEstimation3dPrediction

AVG_PERSON_HEIGHT = 180

# pelvis (body center) is missing, id == 2
map_id_to_panoptic = [1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18]

limbs = [[18, 17, 1],
         [16, 15, 1],
         [5, 4, 3],
         [8, 7, 6],
         [11, 10, 9],
         [14, 13, 12]]


class HumanPose3dAdapter(Adapter):
    __provider__ = 'human_pose_estimation_3d'
    prediction_types = (PoseEstimation3dPrediction,)

    def __init__(self, launcher_config, label_map=None, output_blob=None):
        super().__init__(launcher_config, label_map, output_blob)
        launcher_2d_config = OrderedDict([
            ('type', HumanPoseAdapter.__provider__),
            ('part_affinity_fields_out', launcher_config['part_affinity_fields_out']),
            ('keypoints_heatmap_out', launcher_config['keypoints_heatmap_out'])])
        self.pose_adapter = HumanPoseAdapter(launcher_2d_config, label_map, output_blob)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'features_3d_out': StringField(description="Name of output layer with 3D features."),
            'keypoints_heatmap_out': StringField(description="Name of output layer with keypoints heatmaps."),
            'part_affinity_fields_out': StringField(
                description="Name of output layer with keypoints pairwise relations (part affinity fields)."
            )
        })

        return parameters

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT
        )

    def configure(self):
        self.features_3d = self.get_value_from_config('features_3d_out')
        self.part_affinity_fields = self.get_value_from_config('part_affinity_fields_out')
        self.keypoints_heatmap = self.get_value_from_config('keypoints_heatmap_out')

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        raw_output = zip(
            identifiers, raw_outputs[self.features_3d], raw_outputs[self.keypoints_heatmap],
            raw_outputs[self.part_affinity_fields], frame_meta
        )
        for identifier, features, heatmap, paf, meta in raw_output:
            poses_2d = self._extract_poses_2d(heatmap, paf, meta, identifier)
            if poses_2d.size == 0:
                result.append(PoseEstimation3dPrediction(identifier, poses_2d.x_values, poses_2d.y_values))
                continue

            height, width, _ = meta['image_size']
            scale_y = height / features.shape[1]
            scale_x = width / features.shape[2]
            panoptic_poses_3d, translations, panoptic_poses_2d = HumanPose3dAdapter._parse_poses(
                features, poses_2d, scale_y, scale_x, 1 / scale_x
            )
            if panoptic_poses_2d.size:
                frame_result = PoseEstimation3dPrediction(
                    identifier, panoptic_poses_2d[:, 0:-1:3], panoptic_poses_2d[:, 1:-1:3],
                    panoptic_poses_2d[:, 2:-1:3], panoptic_poses_2d[:, -1], x_3d_values=panoptic_poses_3d[:, 0::4],
                    y_3d_values=panoptic_poses_3d[:, 1::4], z_3d_values=panoptic_poses_3d[:, 2::4],
                    translations=translations
                )
            else:
                frame_result = PoseEstimation3dPrediction(
                    identifier, panoptic_poses_2d, panoptic_poses_2d,
                    panoptic_poses_2d,
                    panoptic_poses_2d, x_3d_values=panoptic_poses_3d,
                    y_3d_values=panoptic_poses_3d, z_3d_values=panoptic_poses_3d,
                    translations=translations
                )
            result.append(frame_result)

        return result

    def _extract_poses_2d(self, heatmap, paf, meta, identifier):
        height, width, _ = meta['image_size']
        heatmap_avg = np.zeros((height, width, 19), dtype=np.float32)
        paf_avg = np.zeros((height, width, 38), dtype=np.float32)
        pad = meta.get('padding', [0, 0, 0, 0])

        heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))
        heatmap = cv2.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[pad[0]:heatmap.shape[0] - pad[2], pad[1]:heatmap.shape[1] - pad[3]:, :]
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap

        paf = np.transpose(np.squeeze(paf), (1, 2, 0))
        paf = cv2.resize(paf, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        paf = paf[pad[0]:paf.shape[0] - pad[2], pad[1]:paf.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
        paf_avg = paf_avg + paf

        peak_counter = 0
        all_peaks = []
        for part in range(0, 18):  # 19th for bg
            peak_counter += self.pose_adapter.find_peaks(heatmap_avg[:, :, part], all_peaks, peak_counter)

        subset, candidate = self.pose_adapter.group_peaks(all_peaks, paf_avg)
        poses_2d = PoseEstimationPrediction(identifier, *HumanPose3dAdapter._get_poses(subset, candidate))
        return poses_2d

    @staticmethod
    def _get_poses(subset, candidate):
        persons_keypoints_x, persons_keypoints_y, persons_keypoints_v = [], [], []
        scores = []
        num_kpt = 18
        for subset_element in subset:
            if subset_element.size == 0:
                continue
            keypoints_x, keypoints_y, keypoints_v = [0] * num_kpt, [0] * num_kpt, [0] * num_kpt
            person_score = subset_element[-2]
            position_id = -1
            for keypoint_id in subset_element[:-2]:
                position_id += 1

                cx, cy, visibility = 0, 0, 0  # Keypoint not found
                if keypoint_id != -1:
                    cx, cy = candidate[keypoint_id.astype(int), 0:2]
                    cx = cx - 0.5 + 1  # +1 for matlab consistency, coords start from 1
                    cy = cy - 0.5 + 1
                    visibility = 1
                keypoints_x[position_id] = cx
                keypoints_y[position_id] = cy
                keypoints_v[position_id] = visibility

            scores.append(person_score * max(0, (subset_element[-1] - 1)))  # -1 for Neck
            persons_keypoints_x.append(keypoints_x)
            persons_keypoints_y.append(keypoints_y)
            persons_keypoints_v.append(keypoints_v)

        persons_keypoints_x = np.array(persons_keypoints_x)
        persons_keypoints_y = np.array(persons_keypoints_y)
        persons_keypoints_v = np.array(persons_keypoints_v)
        scores = np.array(scores)

        return persons_keypoints_x, persons_keypoints_y, persons_keypoints_v, scores

    @staticmethod
    def _get_root_relative_poses(features, found_poses_2d):
        poses_2d = []
        num_kpt_panoptic = 19
        num_kpt = 18
        for pose_id in range(found_poses_2d.size):
            if found_poses_2d.visibility[pose_id, 1] == 0:  # skip pose if does not found neck
                continue
            # just reordering
            pose_2d = np.ones(num_kpt_panoptic * 3 + 1, dtype=np.float32) * -1  # +1 for pose confidence
            for kpt_id in range(num_kpt):
                if found_poses_2d.visibility[pose_id, kpt_id] != 0:
                    pose_2d[map_id_to_panoptic[kpt_id] * 3] = found_poses_2d.x_values[pose_id, kpt_id]
                    pose_2d[map_id_to_panoptic[kpt_id] * 3 + 1] = found_poses_2d.y_values[pose_id, kpt_id]
                    pose_2d[map_id_to_panoptic[kpt_id] * 3 + 2] = found_poses_2d.visibility[pose_id, kpt_id]
            pose_2d[-1] = found_poses_2d.scores[pose_id]
            poses_2d.append(pose_2d)
        poses_2d = np.array(poses_2d)

        keypoint_threshold = 0.1
        poses_3d = np.ones((poses_2d.shape[0], num_kpt_panoptic * 4), dtype=np.float32) * -1
        for pose_id in range(poses_3d.shape[0]):
            pose_3d = poses_3d[pose_id]
            if poses_2d[pose_id, 2] <= keypoint_threshold:
                continue
            neck_2d = poses_2d[pose_id, 0:2].astype(np.int32)
            # read all pose coordinates at neck location
            for kpt_id in range(num_kpt_panoptic):
                map_3d = features[kpt_id * 3:(kpt_id + 1) * 3]
                pose_3d[kpt_id * 4] = map_3d[0, neck_2d[1], neck_2d[0]]
                pose_3d[kpt_id * 4 + 1] = map_3d[1, neck_2d[1], neck_2d[0]]
                pose_3d[kpt_id * 4 + 2] = map_3d[2, neck_2d[1], neck_2d[0]]
                pose_3d[kpt_id * 4 + 3] = poses_2d[pose_id, kpt_id * 3 + 2]

            # refine keypoints coordinates at corresponding limbs locations
            for limb in limbs:
                for kpt_id_from in limb:
                    if poses_2d[pose_id, kpt_id_from * 3 + 2] <= keypoint_threshold:
                        continue
                    for kpt_id_where in limb:
                        kpt_from_2d = poses_2d[pose_id, kpt_id_from * 3: kpt_id_from * 3 + 2].astype(np.int32)
                        map_3d = features[kpt_id_where * 3:(kpt_id_where + 1) * 3]
                        pose_3d[kpt_id_where * 4] = map_3d[0, kpt_from_2d[1], kpt_from_2d[0]]
                        pose_3d[kpt_id_where * 4 + 1] = map_3d[1, kpt_from_2d[1], kpt_from_2d[0]]
                        pose_3d[kpt_id_where * 4 + 2] = map_3d[2, kpt_from_2d[1], kpt_from_2d[0]]
                    break

        poses_3d[:, 0::4] *= AVG_PERSON_HEIGHT
        poses_3d[:, 1::4] *= AVG_PERSON_HEIGHT
        poses_3d[:, 2::4] *= AVG_PERSON_HEIGHT
        return poses_3d, poses_2d

    @staticmethod
    def _parse_poses(features, found_poses_2d, scale_y, scale_x, fx):
        # map 2d coordinates from image to features space
        found_poses_2d.x_values[found_poses_2d.visibility > 0] /= scale_x
        found_poses_2d.y_values[found_poses_2d.visibility > 0] /= scale_y
        poses_3d, poses_2d = HumanPose3dAdapter._get_root_relative_poses(features, found_poses_2d)

        features_shape = features.shape
        translations = []
        # calculate translations
        for pose_id in range(poses_3d.shape[0]):
            pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
            pose_2d = poses_2d[pose_id][:-1].reshape((-1, 3)).transpose()
            num_valid = np.count_nonzero(pose_2d[2] != -1)
            pose_3d_valid = np.zeros((3, num_valid), dtype=np.float32)
            pose_2d_valid = np.zeros((2, num_valid), dtype=np.float32)
            valid_id = 0
            for kpt_id in range(pose_3d.shape[1]):
                if pose_2d[2, kpt_id] == -1:
                    continue
                pose_3d_valid[:, valid_id] = pose_3d[0:3, kpt_id]
                pose_2d_valid[:, valid_id] = pose_2d[0:2, kpt_id]
                valid_id += 1
            assert valid_id == num_valid

            pose_2d_valid[0] = pose_2d_valid[0] - features_shape[2] / 2
            pose_2d_valid[1] = pose_2d_valid[1] - features_shape[1] / 2
            mean_3d = np.expand_dims(pose_3d_valid.mean(axis=1), axis=1)
            mean_2d = np.expand_dims(pose_2d_valid.mean(axis=1), axis=1)
            numerator = np.trace(np.dot((pose_3d_valid[:2, :] - mean_3d[:2, :]).transpose(),
                                        pose_3d_valid[:2, :] - mean_3d[:2, :])).sum()
            numerator = np.sqrt(numerator)
            denominator = np.sqrt(np.trace(np.dot((pose_2d_valid[:2, :] - mean_2d[:2, :]).transpose(),
                                                  pose_2d_valid[:2, :] - mean_2d[:2, :])).sum())
            mean_2d = np.array([mean_2d[0, 0], mean_2d[1, 0], fx])
            mean_3d = np.array([mean_3d[0, 0], mean_3d[1, 0], 0])
            translation = numerator / denominator * mean_2d - mean_3d
            translations.append(translation)

        # map 2d coordinates back to image space
        if poses_2d.size:
            poses_2d[:, 0:-1:3] *= scale_x
            poses_2d[:, 1:-1:3] *= scale_y
        return poses_3d, np.array(translations, dtype=np.float32), poses_2d
