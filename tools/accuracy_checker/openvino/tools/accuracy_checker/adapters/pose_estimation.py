"""
Copyright (c) 2018-2022 Intel Corporation

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

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, ConfigError
from ..preprocessor import ObjectCropWithScale
from ..representation import PoseEstimationPrediction
from ..utils import contains_any


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

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT
        )

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
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.outputs_verified = True
        if self.concat_out:
            self.output_blob = self.check_output_name(self.output_blob, outputs)
            return
        self.part_affinity_fields = self.check_output_name(self.part_affinity_fields, outputs)
        self.keypoints_heatmap = self.check_output_name(self.keypoints_heatmap, outputs)
        self._keypoints_heatmap_bias = self.check_output_name(self._keypoints_heatmap_bias, outputs)
        self._part_affinity_fields_bias = self.check_output_name(self._part_affinity_fields_bias, outputs)

    def process(self, raw, identifiers, frame_meta):
        result = []
        if not self.outputs_verified:
            self.select_output_blob(raw)
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


class SingleHumanPoseAdapter(Adapter):
    __provider__ = 'single_human_pose_estimation'
    prediction_types = (PoseEstimationPrediction, )

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT
        )

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_outputs)
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
        self.score_map_out = self.get_value_from_config('score_map_output')
        self.output_verified = False

    def select_output_blob(self, outputs):
        self.output_verified = True
        if self.score_map_out:
            self.score_map_out = self.check_output_name(self.score_map_out, outputs)
            return
        super().select_output_blob(outputs)
        self.score_map_out = self.output_blob
        return

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not self.output_verified:
            self.select_output_blob(raw_outputs)
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
            if 1 < px < res[0] and 1 < py < res[1]:
                diff = np.array([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                coords[p] += np.sign(diff).astype(float) * .25
        coords += 0.5

        # Transform back
        preds = transform_preds(coords, center, scale, res)

        if preds.size < 3:
            preds = preds.reshape(1, preds.size)

        return preds
