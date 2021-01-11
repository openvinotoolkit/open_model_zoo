"""
Copyright (c) 2020 Intel Corporation

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

try:
    from numpy.core.umath import clip
except ImportError:
    from numpy import clip

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, ConfigError, NumberField
from ..representation import PoseEstimationPrediction
from ..utils import contains_any, UnsupportedPackage

try:
    from skimage.measure import block_reduce
except ImportError as import_error:
    block_reduce = UnsupportedPackage('skimage.measure', import_error.msg)


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

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT
        )

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
        if isinstance(block_reduce, UnsupportedPackage):
            block_reduce.raise_error(self.__provider__)
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
                heatmap = cv2.resize(heatmap, (0, 0), fx=self.upscale_factor, fy=self.upscale_factor,
                                     interpolation=cv2.INTER_CUBIC)
                heatmap = np.transpose(heatmap, (2, 0, 1))
                paf = np.transpose(np.squeeze(paf), (1, 2, 0))
                paf = cv2.resize(paf, (0, 0), fx=self.upscale_factor, fy=self.upscale_factor,
                                 interpolation=cv2.INTER_CUBIC)
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
                clip(kpts[:, 0], 0, w - 1, out=kpts[:, 0])
                clip(kpts[:, 1], 0, h - 1, out=kpts[:, 1])

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
            clip(x, 0, w - 1, out=x)
            clip(y, 0, h - 1, out=y)
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

    @staticmethod
    def refine(heatmap, x, y):
        h, w = heatmap.shape[-2:]
        valid = np.logical_and(np.logical_and(x > 0, x < w - 1), np.logical_and(y > 0, y < h - 1))
        xx = x[valid]
        yy = y[valid]
        dx = np.sign(heatmap[yy, xx + 1] - heatmap[yy, xx - 1], dtype=np.float32) * 0.25
        dy = np.sign(heatmap[yy + 1, xx] - heatmap[yy - 1, xx], dtype=np.float32) * 0.25
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x[valid] += dx
        y[valid] += dy
        return x, y

    @staticmethod
    def is_disjoint(pose_a, pose_b):
        pose_a = pose_a[:-2]
        pose_b = pose_b[:-2]
        return np.all(np.logical_or.reduce((pose_a == pose_b, pose_a < 0, pose_b < 0)))

    def update_poses(self, kpt_a_id, kpt_b_id, all_keypoints, connections, pose_entries, pose_entry_size):
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
                pose_entry = np.full(pose_entry_size, -1, dtype=np.float32)
                pose_entry[kpt_a_id] = connection[0]
                pose_entry[kpt_b_id] = connection[1]
                pose_entry[-1] = 2
                pose_entry[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
                pose_entries.append(pose_entry)
            elif pose_a_idx >= 0 and pose_b_idx >= 0 and pose_a_idx != pose_b_idx:
                # Merge two poses are disjoint merge them, otherwise ignore connection.
                pose_a = pose_entries[pose_a_idx]
                pose_b = pose_entries[pose_b_idx]
                if self.is_disjoint(pose_a, pose_b):
                    pose_a += pose_b
                    pose_a[:-2] += 1
                    pose_a[-2] += connection[2]
                    del pose_entries[pose_b_idx]
            elif pose_a_idx >= 0 and pose_b_idx >= 0:
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

    @staticmethod
    def connections_nms(a_idx, b_idx, affinity_scores):
        # From all retrieved connections that share starting/ending keypoints leave only the top-scoring ones.
        order = affinity_scores.argsort()[::-1]
        affinity_scores = affinity_scores[order]
        a_idx = a_idx[order]
        b_idx = b_idx[order]
        idx = []
        has_kpt_a = set()
        has_kpt_b = set()
        for t, (i, j) in enumerate(zip(a_idx, b_idx)):
            if i not in has_kpt_a and j not in has_kpt_b:
                idx.append(t)
                has_kpt_a.add(i)
                has_kpt_b.add(j)
        idx = np.asarray(idx, dtype=np.int32)
        return a_idx[idx], b_idx[idx], affinity_scores[idx]

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
            if len(valid_limbs) == 0:
                continue
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity_scores = affinity_scores[valid_limbs]

            # Suppress incompatible connections.
            a_idx, b_idx, affinity_scores = self.connections_nms(a_idx, b_idx, affinity_scores)
            connections = list(zip(kpts_a[a_idx, 3].astype(np.int32),
                                   kpts_b[b_idx, 3].astype(np.int32),
                                   affinity_scores))
            if len(connections) == 0:
                continue

            # Update poses with new connections.
            pose_entries = self.update_poses(kpt_a_id, kpt_b_id, all_keypoints,
                                             connections, pose_entries, pose_entry_size)

        # Remove poses with not enough points.
        pose_entries = np.asarray(pose_entries, dtype=np.float32).reshape(-1, pose_entry_size)
        pose_entries = pose_entries[pose_entries[:, -1] >= 3]
        return pose_entries, all_keypoints

    @staticmethod
    def convert_to_coco_format(pose_entries, all_keypoints):
        num_joints = 17
        coco_keypoints = []
        scores = []
        for pose in pose_entries:
            if len(pose) == 0:
                continue
            keypoints = np.zeros(num_joints * 3)
            reorder_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            person_score = pose[-2]
            for keypoint_id, target_id in zip(pose[:-2], reorder_map):
                if target_id < 0:
                    continue
                cx, cy, score = 0, 0, 0  # keypoint not found
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                keypoints[target_id * 3 + 0] = cx
                keypoints[target_id * 3 + 1] = cy
                keypoints[target_id * 3 + 2] = score
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose[-1] - 1)))  # -1 for 'neck'
        return np.asarray(coco_keypoints), np.asarray(scores)
