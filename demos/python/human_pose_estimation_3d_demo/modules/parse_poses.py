#!/usr/bin/env python
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

from modules.pose import Pose, propagate_ids
from pose_extractor import extract_poses

AVG_PERSON_HEIGHT = 180

# pelvis (body center) is missing, id == 2
map_id_to_panoptic = [1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18]

limbs = [[18, 17, 1],
         [16, 15, 1],
         [5, 4, 3],
         [8, 7, 6],
         [11, 10, 9],
         [14, 13, 12]]


def get_root_relative_poses(inference_results):
    features, heatmap, paf_map = inference_results

    upsample_ratio = 4
    found_poses = extract_poses(heatmap[0:-1], paf_map, upsample_ratio)
    # scale coordinates to features space
    found_poses[:, 0:-1:3] /= upsample_ratio
    found_poses[:, 1:-1:3] /= upsample_ratio

    poses_2d = []
    num_kpt_panoptic = 19
    num_kpt = 18
    for pose_id in range(found_poses.shape[0]):
        if found_poses[pose_id, 5] == -1:  # skip pose if does not found neck
            continue
        pose_2d = np.ones(num_kpt_panoptic * 3 + 1, dtype=np.float32) * -1  # +1 for pose confidence
        for kpt_id in range(num_kpt):
            if found_poses[pose_id, kpt_id * 3] != -1:
                x_2d, y_2d, conf = found_poses[pose_id, kpt_id * 3:(kpt_id + 1) * 3]
                pose_2d[map_id_to_panoptic[kpt_id] * 3] = x_2d  # just repacking
                pose_2d[map_id_to_panoptic[kpt_id] * 3 + 1] = y_2d
                pose_2d[map_id_to_panoptic[kpt_id] * 3 + 2] = conf
        pose_2d[-1] = found_poses[pose_id, -1]
        poses_2d.append(pose_2d)
    poses_2d = np.array(poses_2d)

    keypoint_treshold = 0.1
    poses_3d = np.ones((len(poses_2d), num_kpt_panoptic * 4), dtype=np.float32) * -1
    for pose_id in range(poses_3d.shape[0]):
        if poses_2d[pose_id, 2] <= keypoint_treshold:
            continue
        pose_3d = poses_3d[pose_id]
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
                if poses_2d[pose_id, kpt_id_from * 3 + 2] <= keypoint_treshold:
                    continue
                for kpt_id_where in limb:
                    kpt_from_2d = poses_2d[pose_id, kpt_id_from * 3:kpt_id_from * 3 + 2].astype(np.int32)
                    map_3d = features[kpt_id_where * 3:(kpt_id_where + 1) * 3]
                    pose_3d[kpt_id_where * 4] = map_3d[0, kpt_from_2d[1], kpt_from_2d[0]]
                    pose_3d[kpt_id_where * 4 + 1] = map_3d[1, kpt_from_2d[1], kpt_from_2d[0]]
                    pose_3d[kpt_id_where * 4 + 2] = map_3d[2, kpt_from_2d[1], kpt_from_2d[0]]
                break

    poses_3d[:, 0::4] *= AVG_PERSON_HEIGHT
    poses_3d[:, 1::4] *= AVG_PERSON_HEIGHT
    poses_3d[:, 2::4] *= AVG_PERSON_HEIGHT
    return poses_3d, poses_2d


previous_poses_2d = []


def parse_poses(inference_results, input_scale, stride, fx, is_video=False):
    global previous_poses_2d
    features = inference_results[0]
    poses_3d, poses_2d = get_root_relative_poses(inference_results)
    poses_2d_scaled = []
    for pose_2d in poses_2d:
        num_kpt = (pose_2d.shape[0] - 1) // 3
        pose_2d_scaled = np.ones(pose_2d.shape[0], dtype=np.float32) * -1
        for kpt_id in range(num_kpt):
            if pose_2d[kpt_id * 3] != -1:
                pose_2d_scaled[kpt_id * 3] = pose_2d[kpt_id * 3] * stride / input_scale
                pose_2d_scaled[kpt_id * 3 + 1] = pose_2d[kpt_id * 3 + 1] * stride / input_scale
                pose_2d_scaled[kpt_id * 3 + 2] = pose_2d[kpt_id * 3 + 2]
        pose_2d_scaled[-1] = pose_2d[-1]
        poses_2d_scaled.append(pose_2d_scaled)

    if is_video:  # track poses ids
        current_poses_2d = []
        for pose_2d_scaled in poses_2d_scaled:
            pose_keypoints = np.ones((Pose.num_kpts, 2), dtype=np.int32) * -1
            for kpt_id in range(Pose.num_kpts):
                if pose_2d_scaled[kpt_id * 3] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0:2] = pose_2d_scaled[kpt_id * 3:kpt_id * 3 + 2].astype(np.int32)
            pose = Pose(pose_keypoints, pose_2d_scaled[-1])
            current_poses_2d.append(pose)
        propagate_ids(previous_poses_2d, current_poses_2d)
        previous_poses_2d = current_poses_2d

    translated_poses_3d = []
    # translate poses
    for pose_id in range(poses_3d.shape[0]):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_2d = poses_2d[pose_id][0:-1].reshape((-1, 3)).transpose()
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

        pose_2d_valid[0] = pose_2d_valid[0] - features.shape[2]/2
        pose_2d_valid[1] = pose_2d_valid[1] - features.shape[1]/2
        mean_3d = np.expand_dims(pose_3d_valid.mean(axis=1), axis=1)
        mean_2d = np.expand_dims(pose_2d_valid.mean(axis=1), axis=1)
        numerator = np.trace(np.dot((pose_3d_valid[0:2] - mean_3d[0:2]).transpose(),
                                    pose_3d_valid[0:2] - mean_3d[0:2])).sum()
        numerator = np.sqrt(numerator)
        denominator = np.sqrt(np.trace(np.dot((pose_2d_valid[0:2] - mean_2d[0:2]).transpose(),
                                              pose_2d_valid[0:2] - mean_2d[0:2])).sum())
        mean_2d = np.array([mean_2d[0, 0], mean_2d[1, 0], fx * input_scale / stride])
        mean_3d = np.array([mean_3d[0, 0], mean_3d[1, 0], 0])
        translation = numerator / denominator * mean_2d - mean_3d

        if is_video:
            translation = current_poses_2d[pose_id].filter(translation)
        for kpt_id in range(19):
            pose_3d[0, kpt_id] = pose_3d[0, kpt_id] + translation[0]
            pose_3d[1, kpt_id] = pose_3d[1, kpt_id] + translation[1]
            pose_3d[2, kpt_id] = pose_3d[2, kpt_id] + translation[2]
        translated_poses_3d.append(pose_3d.transpose().reshape(-1))

    return np.array(translated_poses_3d), np.array(poses_2d_scaled)
