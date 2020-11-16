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

import math
from operator import itemgetter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import block_reduce

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, ConfigError, NumberField
from ..preprocessor import ObjectCropWithScale
from ..representation import PoseEstimationPrediction
from ..utils import contains_all, contains_any


class HumanPoseAdapter(Adapter):
    __provider__ = 'human_pose_estimation'
    prediction_types = (PoseEstimationPrediction, )

    limb_seq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13],
        [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]
    ]
    map_idx = [
        [31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26],
        [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]
    ]

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'part_affinity_fields_out': StringField(
                description="Name of output layer with keypoints pairwise relations (part affinity fields).",
                optional=True
            ),
            'keypoints_heatmap_out': StringField(
                description="Name of output layer with keypoints heatmaps.", optional=True
            ),
        })

        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.part_affinity_fields = self.get_value_from_config('part_affinity_fields_out')
        self.keypoints_heatmap = self.get_value_from_config('keypoints_heatmap_out')
        self.concat_out = self.part_affinity_fields is None and self.keypoints_heatmap is None
        if not self.concat_out:
            contains_both = self.part_affinity_fields is not None and self.keypoints_heatmap is not None
            if not contains_both:
                raise ConfigError(
                    'human_pose_estimation adapter should contains both: keypoints_heatmap_out '
                    'and part_affinity_fields_out or not contain them at all (in single output model case)'
                )
            self._keypoints_heatmap_bias = self.keypoints_heatmap + '/add_'
            self._part_affinity_fields_bias = self.part_affinity_fields + '/add_'

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.concat_out:
            if not contains_any(raw_outputs, [self.part_affinity_fields, self._part_affinity_fields_bias]):
                raise ConfigError('part affinity fields output not found')
            if not contains_any(raw_outputs, [self.keypoints_heatmap, self._keypoints_heatmap_bias]):
                raise ConfigError('keypoints heatmap output not found')
            keypoints_heatmap = raw_outputs[
                self.keypoints_heatmap if self.keypoints_heatmap in raw_outputs else self._keypoints_heatmap_bias
            ]
            pafs = raw_outputs[
                self.part_affinity_fields if self.part_affinity_fields in raw_outputs
                else self._part_affinity_fields_bias
            ]
            raw_output = zip(identifiers, keypoints_heatmap, pafs, frame_meta)
        else:
            concat_out = raw_outputs[self.output_blob]
            keypoints_num = concat_out.shape[1] // 3
            keypoints_heat_map = concat_out[:, :keypoints_num, :]
            pafs = concat_out[:, keypoints_num:, :]
            raw_output = zip(identifiers, keypoints_heat_map, pafs, frame_meta)
        for identifier, heatmap, paf, meta in raw_output:
            height, width, _ = meta['image_size']
            heatmap_avg = np.zeros((height, width, 19), dtype=np.float32)
            paf_avg = np.zeros((height, width, 38), dtype=np.float32)
            pad = meta.get('padding', [0, 0, 0, 0])
            transpose_order = (1, 2, 0) if heatmap.shape[0] == 19 else (0, 1, 2)

            heatmap = np.transpose(np.squeeze(heatmap), transpose_order)
            heatmap = cv2.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[pad[0]:heatmap.shape[0] - pad[2], pad[1]:heatmap.shape[1] - pad[3]:, :]
            heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
            heatmap_avg = heatmap_avg + heatmap

            paf = np.transpose(np.squeeze(paf), transpose_order)
            paf = cv2.resize(paf, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            paf = paf[pad[0]:paf.shape[0] - pad[2], pad[1]:paf.shape[1] - pad[3], :]
            paf = cv2.resize(paf, (width, height), interpolation=cv2.INTER_CUBIC)
            paf_avg = paf_avg + paf

            peak_counter = 0
            all_peaks = []
            for part in range(0, 18):  # 19th for bg
                peak_counter += self.find_peaks(heatmap_avg[:, :, part], all_peaks, peak_counter)

            subset, candidate = self.group_peaks(all_peaks, paf_avg)
            result.append(PoseEstimationPrediction(identifier, *self.get_poses(subset, candidate)))

        return result

    @staticmethod
    def find_peaks(heatmap, all_peaks, prev_peak_counter):
        heatmap[heatmap < 0.1] = 0
        heatmap[np.isnan(heatmap)] = 0
        map_aug = np.zeros((heatmap.shape[0] + 2, heatmap.shape[1] + 2))
        map_left = np.zeros(map_aug.shape)
        map_right = np.zeros(map_aug.shape)
        map_up = np.zeros(map_aug.shape)
        map_down = np.zeros(map_aug.shape)

        map_aug[1:map_aug.shape[0] - 1, 1:map_aug.shape[1] - 1] = heatmap
        map_left[1:map_aug.shape[0] - 1, :map_aug.shape[1] - 2] = heatmap
        map_right[1:map_aug.shape[0] - 1, 2:map_aug.shape[1]] = heatmap
        map_up[:map_aug.shape[0] - 2, 1:map_aug.shape[1] - 1] = heatmap
        map_down[2:map_aug.shape[0], 1:map_aug.shape[1] - 1] = heatmap

        peaks_binary = (map_aug > map_left) & (map_aug > map_right) & (map_aug > map_up) & (map_aug > map_down)
        peaks_binary = peaks_binary[1:map_aug.shape[0] - 1, 1:map_aug.shape[1] - 1]
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        peaks = sorted(peaks, key=itemgetter(0))  # same order with matlab

        flag = np.ones(len(peaks), np.uint8)
        peaks_with_score_and_id = []
        peak_counter = 0
        for i, _ in enumerate(peaks):
            if flag[i] != 1:
                continue
            for j in range(i + 1, len(peaks)):
                if math.sqrt((peaks[i][0] - peaks[j][0]) ** 2 + (peaks[i][1] - peaks[j][1]) ** 2) < 6:
                    flag[j] = 0
            peak_id = peak_counter + prev_peak_counter
            peak_counter += 1
            peaks_with_score_and_id.append([peaks[i][0], peaks[i][1], heatmap[peaks[i][1], peaks[i][0]], peak_id])
        all_peaks.append(peaks_with_score_and_id)

        return peak_counter

    @staticmethod
    def _add_pose_single_candidate(subset, candidate, idx_joint, kpt_num=20):
        for joint in candidate:
            num = 0
            for subset_j in subset:  # check if already in some pose, was added as a part of another limb
                if subset_j[idx_joint] == joint[3]:
                    num += 1
                    continue
            if num == 0:
                person_keypoints = np.ones(kpt_num) * -1
                person_keypoints[idx_joint] = joint[3]  # joint idx
                person_keypoints[-1] = 1  # n joints in pose
                person_keypoints[-2] = joint[2]  # pose score
                subset.append(person_keypoints)

        return subset

    @staticmethod
    def _filter_subset(subset):
        filtered_subset = []
        for subset_element in subset:
            if subset_element[-1] < 3 or (subset_element[-2] / subset_element[-1] < 0.2):
                continue
            filtered_subset.append(subset_element)

        return np.asarray(filtered_subset)

    @staticmethod
    def _add_pose_both_candidates(subset, temp, index_a, index_b, candidates, kpt_num=20):
        for i, temp_i in enumerate(temp):
            num = 0
            for j, subset_j in enumerate(subset):
                if subset_j[index_a] == temp_i[0]:
                    subset[j][index_b] = temp[i][1]
                    num += 1
                    subset[j][-1] += 1
                    subset[j][-2] += candidates[temp_i[1], 2] + temp_i[2]
            if num == 0:
                person_keypoints = np.ones(kpt_num) * -1
                person_keypoints[index_a] = temp[i][0]
                person_keypoints[index_b] = temp[i][1]
                person_keypoints[-1] = 2
                person_keypoints[-2] = np.sum(candidates[temp_i[0:2], 2]) + temp_i[2]
                subset.append(person_keypoints)

        return subset

    @staticmethod
    def _copy_temperature_to_subset(subset, temp, index_a, index_b):
        for _, temp_i in enumerate(temp):
            for j, subset_j in enumerate(subset):
                check_subset_a = subset_j[index_a] == temp_i[0] and subset_j[index_b] == -1
                check_subset_b = subset_j[index_b] == temp_i[1] and subset_j[index_a] == -1
                if check_subset_a:
                    subset[j][index_b] = temp_i[1]
                    continue
                if check_subset_b:
                    subset[j][index_a] = temp_i[0]

        return subset

    @staticmethod
    def _get_temperature(cand_a_, cand_b_, score_mid, pafs, threshold=0.05):
        temp_ = []
        for index_a_, cand_a_element in enumerate(cand_a_):
            for index_b_, cand_b_element in enumerate(cand_b_):
                mid_point = [(
                    int(round((cand_a_element[0] + cand_b_element[0]) * 0.5)),
                    int(round((cand_a_element[1] + cand_b_element[1]) * 0.5))
                )] * 2
                vec = [cand_b_element[0] - cand_a_element[0], cand_b_element[1] - cand_a_element[1]]
                norm_vec = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                if norm_vec == 0:
                    continue
                vec[0] /= norm_vec
                vec[1] /= norm_vec
                score_mid_a = score_mid[mid_point[0][1], mid_point[0][0], 0]
                score_mid_b = score_mid[mid_point[1][1], mid_point[1][0], 1]
                score = vec[0] * score_mid_a + vec[1] * score_mid_b

                height_n = pafs.shape[0] // 2
                suc_ratio = 0
                mid_score = 0
                mid_num = 10  # n points for integral over paf

                if score > -100:
                    p_sum = 0
                    p_count = 0

                    x = np.linspace(cand_a_element[0], cand_b_element[0], mid_num)
                    y = np.linspace(cand_a_element[1], cand_b_element[1], mid_num)
                    for point_idx in range(0, mid_num):
                        px = int(round(x[point_idx]))
                        py = int(round(y[point_idx]))
                        pred = score_mid[py, px, 0:2]
                        score = vec[0] * pred[0] + vec[1] * pred[1]
                        if score > threshold:
                            p_sum += score
                            p_count += 1
                    suc_ratio = p_count / mid_num
                    ratio = 0
                    if p_count > 0:
                        ratio = p_sum / p_count
                    mid_score = ratio + min(height_n / norm_vec - 1, 0)
                if mid_score > 0 and suc_ratio > 0.8:
                    score = mid_score
                    score_all = score + cand_a_element[2] + cand_b_element[2]
                    temp_.append([index_a_, index_b_, score, score_all])
        if temp_:
            temp_ = sorted(temp_, key=itemgetter(2), reverse=True)

        return temp_

    def _get_connections(self, cand_a, cand_b, score_mid, pafs, thresh):
        temp_ = self._get_temperature(cand_a, cand_b, score_mid, pafs, thresh)
        num_limbs = min(len(cand_a), len(cand_b))
        cnt = 0
        occur_a = np.zeros(len(cand_a), dtype=np.int32)
        occur_b = np.zeros(len(cand_b), dtype=np.int32)
        connections = []
        for row_temp in temp_:
            if cnt == num_limbs:
                break
            i, j, score = row_temp[0:3]
            if occur_a[i] == 0 and occur_b[j] == 0:
                connections.append([cand_a[i][3], cand_b[j][3], score])
                cnt += 1
                occur_a[i] = 1
                occur_b[j] = 1
        return connections

    def group_peaks(self, peaks, pafs, kpt_num=20, threshold=0.05):
        subset = []
        candidates = np.array([item for sublist in peaks for item in sublist])
        for keypoint_id, mapped_keypoints in enumerate(self.map_idx):
            score_mid = pafs[:, :, [x - 19 for x in mapped_keypoints]]
            candidate_a = peaks[self.limb_seq[keypoint_id][0] - 1]
            candidate_b = peaks[self.limb_seq[keypoint_id][1] - 1]
            idx_joint_a = self.limb_seq[keypoint_id][0] - 1
            idx_joint_b = self.limb_seq[keypoint_id][1] - 1

            if not candidate_a and not candidate_b:  # no such limb
                continue
            if not candidate_a:  # limb has just B joint
                subset = self._add_pose_single_candidate(subset, candidate_b, idx_joint_b, kpt_num)
                continue
            if not candidate_b:  # limb has just A joint
                subset = self._add_pose_single_candidate(subset, candidate_a, idx_joint_a, kpt_num)
                continue

            temp = self._get_connections(candidate_a, candidate_b, score_mid, pafs, threshold)
            if not temp:
                continue

            if keypoint_id == 0:
                subset = [np.ones(kpt_num) * -1 for _ in temp]
                for i, temp_i in enumerate(temp):
                    subset[i][self.limb_seq[0][0] - 1] = temp_i[0]
                    subset[i][self.limb_seq[0][1] - 1] = temp_i[1]
                    subset[i][-1] = 2
                    subset[i][-2] = np.sum(candidates[temp_i[0:2], 2]) + temp_i[2]
            else:
                index_a = self.limb_seq[keypoint_id][0] - 1
                index_b = self.limb_seq[keypoint_id][1] - 1
                if keypoint_id in (17, 18):
                    subset = self._copy_temperature_to_subset(subset, temp, index_a, index_b)
                    continue
                subset = self._add_pose_both_candidates(subset, temp, index_a, index_b, candidates, kpt_num)

        return self._filter_subset(subset), candidates

    @staticmethod
    def get_poses(subset, candidate):
        persons_keypoints_x, persons_keypoints_y, persons_keypoints_v = [], [], []
        scores = []
        for subset_element in subset:
            if subset_element.size == 0:
                continue
            keypoints_x, keypoints_y, keypoints_v = [0] * 17, [0] * 17, [0] * 17
            to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            person_score = subset_element[-2]
            position_id = -1
            for keypoint_id in subset_element[:-2]:
                position_id += 1
                if position_id == 1:  # No 'Neck' in COCO
                    continue

                cx, cy, visibility = 0, 0, 0  # Keypoint not found
                if keypoint_id != -1:
                    cx, cy = candidate[keypoint_id.astype(int), 0:2]
                    cx = cx - 0.5 + 1  # +1 for matlab consistency, coords start from 1
                    cy = cy - 0.5 + 1
                    visibility = 1
                keypoints_x[to_coco_map[position_id]] = cx
                keypoints_y[to_coco_map[position_id]] = cy
                keypoints_v[to_coco_map[position_id]] = visibility

            scores.append(person_score * max(0, (subset_element[-1] - 1)))  # -1 for Neck
            persons_keypoints_x.append(keypoints_x)
            persons_keypoints_y.append(keypoints_y)
            persons_keypoints_v.append(keypoints_v)

        persons_keypoints_x = np.array(persons_keypoints_x)
        persons_keypoints_y = np.array(persons_keypoints_y)
        persons_keypoints_v = np.array(persons_keypoints_v)
        scores = np.array(scores)

        return persons_keypoints_x, persons_keypoints_y, persons_keypoints_v, scores


class OpenPoseAdapter(Adapter):
    __provider__ = 'human_pose_estimation_openpose'
    prediction_types = (PoseEstimationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'part_affinity_fields_out': StringField(
                description="Name of output layer with keypoints pairwise relations (part affinity fields).",
                optional=True
            ),
            'keypoints_heatmap_out': StringField(
                description="Name of output layer with keypoints heatmaps.", optional=True
            ),
            'upscale_factor': NumberField(
                description="Upscaling factor for output feature maps before postprocessing.",
                value_type=float, min_value=1, default=1, optional=True
            ),
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.upscale_factor = self.get_value_from_config('upscale_factor')
        self.part_affinity_fields = self.get_value_from_config('part_affinity_fields_out')
        self.keypoints_heatmap = self.get_value_from_config('keypoints_heatmap_out')
        self.concat_out = self.part_affinity_fields is None and self.keypoints_heatmap is None
        if not self.concat_out:
            contains_both = self.part_affinity_fields is not None and self.keypoints_heatmap is not None
            if not contains_both:
                raise ConfigError(
                    'human_pose_estimation adapter should contains both: keypoints_heatmap_out '
                    'and part_affinity_fields_out or not contain them at all (in single output model case)'
                )
            self._keypoints_heatmap_bias = self.keypoints_heatmap + '/add_'
            self._part_affinity_fields_bias = self.part_affinity_fields + '/add_'

        self.decoder = OpenPoseDecoder(num_joints=18, delta=0.5 if self.upscale_factor == 1 else 0.0)
        self.nms = HeatmapNMS(kernel=2 * int(np.round(6 / 7 * self.upscale_factor)) + 1)

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.concat_out:
            if not contains_any(raw_outputs, [self.part_affinity_fields, self._part_affinity_fields_bias]):
                raise ConfigError('part affinity fields output not found')
            if not contains_any(raw_outputs, [self.keypoints_heatmap, self._keypoints_heatmap_bias]):
                raise ConfigError('keypoints heatmap output not found')
            keypoints_heatmap = raw_outputs[
                self.keypoints_heatmap if self.keypoints_heatmap in raw_outputs else self._keypoints_heatmap_bias
            ]
            pafs = raw_outputs[
                self.part_affinity_fields if self.part_affinity_fields in raw_outputs
                else self._part_affinity_fields_bias
            ]
            raw_output = zip(identifiers, keypoints_heatmap, pafs, frame_meta)
        else:
            concat_out = raw_outputs[self.output_blob]
            keypoints_num = concat_out.shape[1] // 3
            keypoints_heat_map = concat_out[:, :keypoints_num, :]
            pafs = concat_out[:, keypoints_num:, :]
            raw_output = zip(identifiers, keypoints_heat_map, pafs, frame_meta)
        for identifier, heatmap, paf, meta in raw_output:
            output_h, output_w = heatmap.shape[-2:]
            if self.upscale_factor > 1:
                self.decoder.delta = 0
                heatmap = np.transpose(heatmap, (1, 2, 0))
                heatmap = cv2.resize(heatmap, (0, 0), fx=self.upscale_factor, fy=self.upscale_factor, interpolation=cv2.INTER_CUBIC)
                heatmap = np.transpose(heatmap, (2, 0, 1))
                paf = np.transpose(np.squeeze(paf), (1, 2, 0))
                paf = cv2.resize(paf, (0, 0), fx=self.upscale_factor, fy=self.upscale_factor, interpolation=cv2.INTER_CUBIC)
                paf = np.transpose(paf, (2, 0, 1))
            hmap = heatmap[None]
            nms_hmap = self.nms(hmap)
            poses, scores = self.decoder(hmap, nms_hmap, paf[None])
            if len(scores) == 0:
                result.append(PoseEstimationPrediction(
                    identifier,
                    np.empty((0, 17), dtype=float),
                    np.empty((0, 17), dtype=float),
                    np.empty((0, 17), dtype=float),
                    np.empty((0, ), dtype=float)
                ))
                continue
            poses = poses.astype(float)
            scores = np.asarray(scores).astype(float)
            scale_x = meta['scale_x']
            scale_y = meta['scale_y']
            input_h, input_w = next(iter(meta['input_shape'].values()))[-2:]
            output_scale_x = input_w / output_w
            output_scale_y = input_h / output_h
            poses[:, :, 0] *= output_scale_x / self.upscale_factor / scale_x
            poses[:, :, 1] *= output_scale_y / self.upscale_factor / scale_y
            point_scores = poses[:, :, 2]

            result.append(PoseEstimationPrediction(
                identifier,
                poses[:, :, 0],
                poses[:, :, 1],
                point_scores,
                scores))
        return result


class AssociativeEmbeddingDecoder:

    def __init__(self, num_joints, max_num_people, detection_threshold, use_detection_val,
                 ignore_too_much, tag_threshold,
                 adjust=True, refine=True, delta=0.0, joints_order=None):
        self.num_joints = num_joints
        self.max_num_people = max_num_people
        self.detection_threshold = detection_threshold
        self.tag_threshold = tag_threshold
        self.use_detection_val = use_detection_val
        self.ignore_too_much = ignore_too_much

        if self.num_joints == 17 and joints_order is None:
            self.joint_order = (0, 1, 2, 3, 4, 5, 6, 11, 12, 7, 8, 9, 10, 13, 14, 15, 16)
        else:
            self.joint_order = list(np.arange(self.num_joints))

        self.do_adjust = adjust
        self.do_refine = refine
        self.delta = delta

    def match(self, tag_k, loc_k, val_k):
        return list(map(self._match_by_tag, zip(tag_k, loc_k, val_k)))

    def _max_match(self, scores):
        r, c = linear_sum_assignment(scores)
        tmp = np.stack((r, c), axis=1)
        return tmp

    def _match_by_tag(self, inp):
        tag_k, loc_k, val_k = inp

        embd_size = tag_k.shape[2]

        class Pose:
            def __init__(self, num_joints, tag_size=1):
                self.num_joints = num_joints
                self.tag_size = tag_size
                self.pose = np.zeros((num_joints, 2 + 1 + tag_size), dtype=np.float32)
                self.pose_tag = np.zeros(tag_size, dtype=np.float32)
                self.valid_points_num = 0

            def add(self, idx, joint, tag):
                self.pose[idx] = joint
                self.pose_tag = (self.pose_tag * self.valid_points_num) + tag
                self.valid_points_num += 1
                self.pose_tag /= self.valid_points_num

            @property
            def tag(self):
                if self.valid_points_num > 0:
                    return self.pose_tag
                else:
                    return None

        all_joints = np.concatenate((loc_k, val_k[..., None], tag_k), -1)

        poses = []
        for idx in self.joint_order:
            tags = tag_k[idx]
            joints = all_joints[idx]
            mask = joints[:, 2] > self.detection_threshold
            tags = tags[mask]
            joints = joints[mask]

            if joints.shape[0] == 0:
                continue

            if len(poses) == 0:
                for tag, joint in zip(tags, joints):
                    pose = Pose(self.num_joints, embd_size)
                    pose.add(idx, joint, tag)
                    poses.append(pose)
            else:
                if self.ignore_too_much and len(poses) == self.max_num_people:
                    continue
                poses_tags = np.stack([p.tag for p in poses], axis=0)

                diff = tags[:, None] - poses_tags[None, :]
                diff_normed = np.linalg.norm(diff, ord=2, axis=2)
                diff_saved = np.copy(diff_normed)

                if self.use_detection_val:
                    diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

                num_added = diff.shape[0]
                num_grouped = diff.shape[1]

                if num_added > num_grouped:
                    diff_normed = np.concatenate(
                        (diff_normed,
                        np.zeros((num_added, num_added - num_grouped), dtype=np.float32) + 1e10),
                        axis=1)

                pairs = self._max_match(diff_normed)
                for row, col in pairs:
                    if row < num_added and col < num_grouped and diff_saved[row][col] < self.tag_threshold:
                        poses[col].add(idx, joints[row], tags[row])
                    else:
                        pose = Pose(self.num_joints, embd_size)
                        pose.add(idx, joints[row], tags[row])
                        poses.append(pose)

        if len(poses):
            ans = np.stack([p.pose for p in poses]).astype(np.float32)
            tags = np.stack([p.tag for p in poses]).astype(np.float32)
        else:
            ans = np.empty((0, self.num_joints, 2 + 1 + embd_size), dtype=np.float32)
            tags = np.empty((0, embd_size), dtype=np.float32)
        return ans, tags

    def top_k(self, heatmaps, tags):
        N, K, H, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        ind = heatmaps.argpartition(-self.max_num_people, axis=2)[:, :, -self.max_num_people:]
        val_k = np.take_along_axis(heatmaps, ind, axis=2)
        subind = np.argsort(-val_k, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        val_k = np.take_along_axis(val_k, subind, axis=2)

        tags = tags.reshape(N, K, W * H, -1)
        tag_k = [np.take_along_axis(tags[..., i], ind, axis=2) for i in range(tags.shape[3])]
        tag_k = np.stack(tag_k, axis=3)

        x = ind % W
        y = ind // W
        ind_k = np.stack((x, y), axis=3)

        ans = {'tag_k': tag_k, 'loc_k': ind_k, 'val_k': val_k}
        return ans

    def adjust(self, ans, heatmaps):
        H, W = heatmaps.shape[-2:]
        for n, people in enumerate(ans):
            for person in people:
                for k, joint in enumerate(person):
                    heatmap = heatmaps[n, k]
                    px = int(joint[0])
                    py = int(joint[1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py, px + 1] - heatmap[py, px - 1],
                            heatmap[py + 1, px] - heatmap[py - 1, px]
                        ])
                        joint[:2] += np.sign(diff) * .25
        return ans

    def refine(self, heatmap, tag, keypoints, pose_tag=None):
        K, H, W = heatmap.shape
        if len(tag.shape) == 3:
            tag = tag[..., None]

        if pose_tag is not None:
            prev_tag = pose_tag
        else:
            tags = []
            for i in range(K):
                if keypoints[i, 2] > 0:
                    x, y = keypoints[i][:2].astype(int)
                    tags.append(tag[i, y, x])
            prev_tag = np.mean(tags, axis=0)

        # Allocate the buffer for tags similarity matrix.
        tag_copy = np.empty_like(tag[0, ..., 0])
        for i, (_heatmap, _tag) in enumerate(zip(heatmap, tag)):
            if keypoints[i, 2] > 0:
                continue
            tag_copy[...] = _tag[..., 0]
            diff = tag_copy
            diff -= prev_tag
            np.abs(diff, out=diff)
            np.floor(diff + 0.5, out=diff)
            diff -= _heatmap
            idx = diff.argmin()
            y, x = np.divmod(idx, _heatmap.shape[-1])

            # detection score at maximum position
            val = _heatmap[y, x]

            if val > 0:
                keypoints[i, :3] = x, y, val
                if 1 < x < W - 1 and 1 < y < H - 1:
                    diff = np.array([
                        _heatmap[y, x + 1] - _heatmap[y, x - 1],
                        _heatmap[y + 1, x] - _heatmap[y - 1, x]
                    ])
                    keypoints[i, :2] += np.sign(diff) * .25

        return keypoints

    def __call__(self, heatmaps, tags, nms_heatmaps=None):
        ans = self.match(**self.top_k(nms_heatmaps, tags))
        ans, ans_tags = map(list, zip(*ans))

        if self.do_adjust:
            ans = self.adjust(ans, heatmaps)

        if self.delta != 0.0:
            for people in ans:
                for person in people:
                    for joint in person:
                        joint[:2] += self.delta

        ans = ans[0]
        scores = np.asarray([i[:, 2].mean() for i in ans])

        if self.do_refine:
            heatmap_numpy = heatmaps[0]
            tag_numpy = tags[0]
            for i in range(len(ans)):
                ans[i] = self.refine(heatmap_numpy, tag_numpy, ans[i], ans_tags[0][i])

        return ans, scores


class AssociativeEmbeddingAdapter(Adapter):
    __provider__ = 'human_pose_estimation_ae'
    prediction_types = (PoseEstimationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'heatmaps_out': StringField(
                description="Name of output layer with keypoints heatmaps.",
                optional=True
            ),
            'nms_heatmaps_out': StringField(
                description="Name of output layer with keypoints heatmaps after NMS.",
                optional=True
            ),
            'embeddings_out': StringField(
                description="Name of output layer with associative embeddings.",
                optional=True
            ),
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.heatmaps = self.get_value_from_config('heatmaps_out')
        self.nms_heatmaps = self.get_value_from_config('nms_heatmaps_out')
        self.embeddings = self.get_value_from_config('embeddings_out')
        self.decoder = AssociativeEmbeddingDecoder(
            num_joints=17,
            adjust=True,
            refine=True,
            delta=0.0,
            max_num_people=30,
            detection_threshold=0.1,
            tag_threshold=1,
            use_detection_val=True,
            ignore_too_much=False)

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not contains_all(raw_outputs, (self.heatmaps, self.nms_heatmaps, self.embeddings)):
            raise ConfigError('Some of the outputs are not found')
        raw_output = zip(identifiers, raw_outputs[self.heatmaps][None],
                         raw_outputs[self.nms_heatmaps][None],
                         raw_outputs[self.embeddings][None], frame_meta)

        for identifier, heatmap, nms_heatmap, embedding, meta in raw_output:
            poses, scores = self.decoder(heatmap, embedding, nms_heatmaps=nms_heatmap)
            if len(scores) == 0:
                result.append(PoseEstimationPrediction(
                    identifier,
                    np.empty((0, 17), dtype=float),
                    np.empty((0, 17), dtype=float),
                    np.empty((0, 17), dtype=float),
                    np.empty((0, ), dtype=float)
                ))
                continue
            poses = poses.astype(float)
            scores = np.asarray(scores).astype(float)
            scale_x = meta['scale_x']
            scale_y = meta['scale_y']
            poses[:, :, 0] /= scale_x / 2
            poses[:, :, 1] /= scale_y / 2
            point_scores = poses[:, :, 2]
            result.append(PoseEstimationPrediction(
                identifier,
                poses[:, :, 0],
                poses[:, :, 1],
                point_scores,
                scores))
        return result


class SingleHumanPoseAdapter(Adapter):
    __provider__ = 'single_human_pose_estimation'
    prediction_types = (PoseEstimationPrediction, )

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT)

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)

        outputs_batch = raw_outputs[self.output_blob]
        for i, heatmaps in enumerate(outputs_batch):
            heatmaps = np.transpose(heatmaps, (1, 2, 0))
            sum_score = 0
            sum_score_thr = 0
            scores = []
            x_values = []
            y_values = []
            num_kp_thr = 0
            vis = [1] * outputs_batch.shape[1]
            for kpt_idx in range(outputs_batch.shape[1]):
                score, coord = self.extract_keypoints(heatmaps[:, :, kpt_idx])
                scores.append(score)
                x, y = self.affine_transform(coord, frame_meta[0]['rev_trans'])
                x_values.append(x)
                y_values.append(y)
                if score > 0.2:
                    sum_score_thr += score
                    num_kp_thr += 1
                sum_score += score
            if num_kp_thr != 0:
                pose_score = sum_score_thr / num_kp_thr
            else:
                pose_score = sum_score / outputs_batch.shape[1]
            result.append(PoseEstimationPrediction(identifiers[i], np.array([x_values]),
                                                   np.array([y_values]), np.array([vis]), np.array([pose_score])))

        return result

    @staticmethod
    def extract_keypoints(heatmap, min_confidence=-100):
        ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
        if heatmap[ind] < min_confidence:
            ind = (-1, -1)
        else:
            ind = (int(ind[1]), int(ind[0]))
        return heatmap[ind[1]][ind[0]], ind

    @staticmethod
    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.])
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]


class StackedHourGlassNetworkAdapter(Adapter):
    __provider__ = 'stacked_hourglass'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({'score_map_output': StringField(optional=True)})
        return params

    def configure(self):
        self.score_map_out = self.get_value_from_config('score_map_out')

    def process(self, raw, identifiers, frame_meta):
        if self.score_map_out is None:
            self.score_map_out = self.output_blob
        raw_outputs = self._extract_predictions(raw, frame_meta)
        score_map_batch = raw_outputs[self.score_map_out]
        result = []
        for identifier, score_map, meta in zip(identifiers, score_map_batch, frame_meta):
            center = meta['center']
            scale = meta['scale']
            points = self.generate_points(score_map, center, scale, [64, 64])
            x_points, y_points = points.T
            result.append(PoseEstimationPrediction(identifier, x_values=x_points, y_values=y_points))
        return result

    @staticmethod
    def generate_points(output, center, scale, res):
        def transform_preds(coords, center, scale, res):
            for p in range(coords.shape[0]):
                coords[p] = ObjectCropWithScale.transform(coords[p], center, scale, res, 1)
            return coords

        def get_preds(scores):
            assert len(scores.shape) == 3, 'Score maps should be 3-dim'

            idx = np.argmax(scores.reshape((scores.shape[0], -1)), 1)

            maxval = np.max(scores.reshape((scores.shape[0], -1)), 1)
            idx = idx.reshape((scores.shape[0], 1)) + 1
            maxval = maxval.reshape((scores.shape[0], 1))

            preds = np.tile(idx, (1, 2))

            preds[:, 0] = (preds[:, 0] - 1) % scores.shape[2] + 1
            preds[:, 1] = np.floor((preds[:, 1] - 1) / scores.shape[2]) + 1

            pred_mask = np.tile(maxval > 0, (1, 2))
            preds *= pred_mask
            return preds

        coords = get_preds(output).astype(float)  # float type

        # pose-processing
        for p in range(coords.shape[0]):
            hm = output[p]
            px = int(math.floor(coords[p][0]))
            py = int(math.floor(coords[p][1]))
            if 1 < px < res[0] and  1 < py < res[1]:
                diff = np.array([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                coords[p] += np.sign(diff).astype(float) * .25
        coords += 0.5

        # Transform back
        preds = transform_preds(coords, center, scale, res)

        if preds.size < 3:
            preds = preds.reshape(1, preds.size)

        return preds


class HeatmapNMS:
    def __init__(self, kernel):
        self.kernel = kernel
        self.pad = (kernel - 1) // 2

    def max_pool(self, x):
        # Max pooling kernel x kernel with stride 1 x 1.
        k = self.kernel
        p = self.pad
        pooled = np.zeros_like(x)
        hmap = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        h, w = hmap.shape[-2:]
        for i in range(k):
            n = (h - i) // k * k
            for j in range(k):
                m = (w - j) // k * k
                hmap_slice = hmap[..., i:i + n, j:j + m]
                pooled[..., i::k, j::k] = block_reduce(hmap_slice, (1, 1, k, k), np.max)
        return pooled

    def __call__(self, heatmaps):
        pooled = self.max_pool(heatmaps)
        return heatmaps * (pooled == heatmaps).astype(heatmaps.dtype)



class OpenPoseDecoder:

    BODY_PARTS_KPT_IDS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
                          (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17))
    BODY_PARTS_PAF_IDS = (12, 20, 14, 16, 22, 24, 0, 2, 4, 6, 8, 10, 28, 30, 34, 32, 36, 18, 26)

    def __init__(self, num_joints=18, skeleton=BODY_PARTS_KPT_IDS, paf_indices=BODY_PARTS_PAF_IDS,
                 max_points=100, score_threshold=0.1, min_paf_alignment_score=0.05, delta=0.5):
        self.num_joints = num_joints
        self.skeleton = skeleton
        self.paf_indices = paf_indices
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.min_paf_alignment_score = min_paf_alignment_score
        self.delta = delta

        self.points_per_limb = 10
        self.grid = np.arange(self.points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    def __call__(self, heatmaps, nms_heatmaps, pafs):
        batch_size, _, h, w = heatmaps.shape
        assert batch_size == 1, 'Batch size of 1 only supported'

        keypoints = self.extract_points(heatmaps, nms_heatmaps)
        pafs = np.transpose(pafs, (0, 2, 3, 1))

        if self.delta > 0:
            for kpts in keypoints:
                kpts[:, :2] += self.delta
                np.core.umath.clip(kpts[:, 0], 0, w - 1, out=kpts[:, 0])
                np.core.umath.clip(kpts[:, 1], 0, h - 1, out=kpts[:, 1])

        pose_entries, keypoints = self.group_keypoints(keypoints, pafs, pose_entry_size=self.num_joints + 2)
        poses, scores = self.convert_to_coco_format(pose_entries, keypoints)
        if len(poses) > 0:
            poses = np.asarray(poses, dtype=np.float32)
            poses = poses.reshape((poses.shape[0], -1, 3))
        else:
            poses = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)

        return poses, scores

    def extract_points(self, heatmaps, nms_heatmaps):
        batch_size, channels_num, h, w = heatmaps.shape
        assert batch_size == 1, 'Batch size of 1 only supported'
        assert channels_num >= self.num_joints

        xs, ys, scores = self.top_k(nms_heatmaps)
        masks = scores > self.score_threshold
        all_keypoints = []
        keypoint_id = 0
        for k in range(self.num_joints):
            # Filter low-score points.
            mask = masks[0, k]
            x = xs[0, k][mask].ravel()
            y = ys[0, k][mask].ravel()
            score = scores[0, k][mask].ravel()
            n = len(x)
            if n == 0:
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                continue
            # Apply quarter offset to improve localization accuracy.
            x, y = self.refine(heatmaps[0, k], x, y)
            np.core.umath.clip(x, 0, w - 1, out=x)
            np.core.umath.clip(y, 0, h - 1, out=y)
            # Pack resulting points.
            keypoints = np.empty((n, 4), dtype=np.float32)
            keypoints[:, 0] = x
            keypoints[:, 1] = y
            keypoints[:, 2] = score
            keypoints[:, 3] = np.arange(keypoint_id, keypoint_id + n)
            keypoint_id += n
            all_keypoints.append(keypoints)
        return all_keypoints

    def top_k(self, heatmaps):
        N, K, _, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        # Get positions with top scores.
        ind = heatmaps.argpartition(-self.max_points, axis=2)[:, :, -self.max_points:]
        scores = np.take_along_axis(heatmaps, ind, axis=2)
        # Keep top scores sorted.
        subind = np.argsort(-scores, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        scores = np.take_along_axis(scores, subind, axis=2)
        y, x = np.divmod(ind, W)
        return x, y, scores

    def refine(self, heatmap, x, y):
        h, w = heatmap.shape[-2:]
        valid = np.logical_and(np.logical_and(0 < x, x < w - 1), np.logical_and(0 < y, y < h - 1))
        xx = x[valid]
        yy = y[valid]
        dx = np.sign(heatmap[yy, xx + 1] - heatmap[yy, xx - 1], dtype=np.float32) * 0.25
        dy = np.sign(heatmap[yy + 1, xx] - heatmap[yy - 1, xx], dtype=np.float32) * 0.25
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x[valid] += dx
        y[valid] += dy
        return x, y

    def update_poses(self, part_id, kpt_a_id, kpt_b_id, all_keypoints, connections, pose_entries, pose_entry_size):
        if part_id == 0:
            pose_entries = [np.full(pose_entry_size, -1, dtype=np.float32) for _ in range(len(connections))]
            for pose, connection in zip(pose_entries, connections):
                pose[kpt_a_id] = connection[0]
                pose[kpt_b_id] = connection[1]
                pose[-1] = 2
                # pose score = sum of all points' scores + sum of all connections' scores
                pose[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
        else:
            for connection in connections:
                pose_a_idx = -1
                pose_b_idx = -1
                for j, pose in enumerate(pose_entries):
                    if pose[kpt_a_id] == connection[0]:
                        pose_a_idx = j
                    if pose[kpt_b_id] == connection[1]:
                        pose_b_idx = j
                if pose_a_idx < 0 and pose_b_idx < 0:
                    # Create new pose entry.
                    pose_entry = np.full(pose_entry_size, -1)
                    pose_entry[kpt_a_id] = connection[0]
                    pose_entry[kpt_b_id] = connection[1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
                    pose_entries.append(pose_entry)
                elif pose_a_idx >= 0 and pose_b_idx >= 0:
                    if pose_a_idx != pose_b_idx:
                        # Merge two disjoint components into one pose.
                        pose_a = pose_entries[pose_a_idx]
                        pose_b = pose_entries[pose_b_idx]
                        do_merge_poses = True
                        for j in range(len(pose_b) - 2):
                            if pose_a[j] >= 0 and pose_b[j] >= 0 and pose_a[j] != pose_b[j]:
                                do_merge_poses = False
                                break
                        if not do_merge_poses:
                            continue
                        for j in range(len(pose_b) - 2):
                            if pose_b[j] >= 0:
                                pose_a[j] = pose_b[j]
                        pose_a[-1] += pose_b[-1]
                        pose_a[-2] += pose_b[-2] + connection[2]
                        del pose_entries[pose_b_idx]
                    else:
                        # Adjust score of a pose.
                        pose_entries[pose_a_idx][-2] += connection[2]
                elif pose_a_idx >= 0:
                    # Add a new limb into pose.
                    pose = pose_entries[pose_a_idx]
                    if pose[kpt_b_id] < 0:
                        pose[-2] += all_keypoints[connection[1], 2]
                    pose[kpt_b_id] = connection[1]
                    pose[-2] += connection[2]
                    pose[-1] += 1
                elif pose_b_idx >= 0:
                    # Add a new limb into pose.
                    pose = pose_entries[pose_b_idx]
                    if pose[kpt_a_id] < 0:
                        pose[-2] += all_keypoints[connection[0], 2]
                    pose[kpt_a_id] = connection[0]
                    pose[-2] += connection[2]
                    pose[-1] += 1
        return pose_entries

    def group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20):
        all_keypoints = np.concatenate(all_keypoints_by_type, axis=0)
        pose_entries = []
        # For every limb.
        for part_id, paf_channel in enumerate(self.paf_indices):
            kpt_a_id, kpt_b_id = self.skeleton[part_id]
            kpts_a = all_keypoints_by_type[kpt_a_id]
            kpts_b = all_keypoints_by_type[kpt_b_id]
            n = len(kpts_a)
            m = len(kpts_b)
            if n == 0 or m == 0:
                continue

            # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
            a = kpts_a[:, :2]
            a = np.broadcast_to(a[None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

            # Sample points along every candidate limb vector.
            steps = (1 / (self.points_per_limb - 1) * vec_raw)
            points = steps * self.grid + a.reshape(-1, 1, 2)
            points = points.round().astype(dtype=np.int32)
            x = points[..., 0].ravel()
            y = points[..., 1].ravel()

            # Compute affinity score between candidate limb vectors and part affinity field.
            part_pafs = pafs[0, :, :, paf_channel:paf_channel + 2]
            field = part_pafs[y, x].reshape(-1, self.points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field * vec).sum(-1).reshape(-1, self.points_per_limb)
            valid_affinity_scores = affinity_scores > self.min_paf_alignment_score
            valid_num = valid_affinity_scores.sum(1)
            affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
            success_ratio = valid_num / self.points_per_limb

            # Get a list of limbs according to the obtained affinity score.
            valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            affinity_scores = affinity_scores[valid_limbs]
            b_idx, a_idx = np.divmod(valid_limbs, n)
            if len(affinity_scores) == 0:
                continue

            # From all retrieved connections that share starting/ending keypoints leave only the top-scoring ones.
            order = affinity_scores.argsort()[::-1]
            affinity_scores = affinity_scores[order]
            a_idx = a_idx[order]
            b_idx = b_idx[order]
            a_idx_unique = np.unique(a_idx, return_index=True)[1]
            b_idx_unique = np.unique(b_idx, return_index=True)[1]
            idx = np.intersect1d(a_idx_unique, b_idx_unique, assume_unique=True)
            a = kpts_a[a_idx[idx], 3].astype(np.int32)
            b = kpts_b[b_idx[idx], 3].astype(np.int32)
            connections = list(zip(a, b, affinity_scores[idx]))

            if len(connections) == 0:
                continue

            # Update poses with new connections.
            pose_entries = self.update_poses(part_id, kpt_a_id, kpt_b_id, all_keypoints,
                                             connections, pose_entries, pose_entry_size)

        # Remove poses with not enough points.
        filtered_entries = []
        for i in range(len(pose_entries)):
            if pose_entries[i][-1] < 3:
                continue
            filtered_entries.append(pose_entries[i])
        pose_entries = np.asarray(filtered_entries, dtype=np.float32)
        return pose_entries, all_keypoints

    def convert_to_coco_format(self, pose_entries, all_keypoints):
        num_joints = 17
        coco_keypoints = []
        scores = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            keypoints = np.zeros(num_joints * 3)
            reorder_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            person_score = pose_entries[n][-2]
            for keypoint_id, target_id in zip(pose_entries[n][:-2], reorder_map):
                if target_id < 0:
                    continue
                cx, cy, score = 0, 0, 0  # keypoint not found
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                keypoints[target_id * 3 + 0] = cx
                keypoints[target_id * 3 + 1] = cy
                keypoints[target_id * 3 + 2] = score
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
        return np.asarray(coco_keypoints), np.asarray(scores)
