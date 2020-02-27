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

import cv2
import numpy as np

from modules.one_euro_filter import OneEuroFilter


class Pose:
    num_kpts = 18
    kpt_names = ['neck', 'nose',
                 'l_sho', 'l_elb', 'l_wri', 'l_hip', 'l_knee', 'l_ank',
                 'r_sho', 'r_elb', 'r_wri', 'r_hip', 'r_knee', 'r_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.79, .26, .79, .72, .62, 1.07, .87, .89, .79, .72, .62, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(keypoints.shape[0]):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        self.bbox = cv2.boundingRect(found_keypoints)
        self.id = None
        self.translation_filter = [OneEuroFilter(freq=80, beta=0.01),
                                   OneEuroFilter(freq=80, beta=0.01),
                                   OneEuroFilter(freq=80, beta=0.01)]

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def filter(self, translation):
        filtered_translation = []
        for coordinate_id in range(3):
            filtered_translation.append(self.translation_filter[coordinate_id](translation[coordinate_id]))
        return filtered_translation


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def propagate_ids(previous_poses, current_poses, threshold=3):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :return: None
    """
    current_poses_sorted_ids = list(range(len(current_poses)))
    current_poses_sorted_ids = sorted(
        current_poses_sorted_ids, key=lambda pose_id: current_poses[pose_id].confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose_id in current_poses_sorted_ids:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for previous_pose_id in range(len(previous_poses)):
            if not mask[previous_pose_id]:
                continue
            iou = get_similarity(current_poses[current_pose_id], previous_poses[previous_pose_id])
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_poses[previous_pose_id].id
                best_matched_id = previous_pose_id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_poses[current_pose_id].update_id(best_matched_pose_id)
        if best_matched_pose_id is not None:
            current_poses[current_pose_id].translation_filter = previous_poses[best_matched_id].translation_filter
