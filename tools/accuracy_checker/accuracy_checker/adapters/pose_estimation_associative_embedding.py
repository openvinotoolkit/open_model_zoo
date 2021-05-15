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

import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, ConfigError
from ..representation import PoseEstimationPrediction
from ..utils import UnsupportedPackage, contains_all

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as error:
    linear_sum_assignment = UnsupportedPackage('scipy.optimize', error.msg)


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

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT
        )

    def configure(self):
        self.heatmaps = self.get_value_from_config('heatmaps_out')
        self.nms_heatmaps = self.get_value_from_config('nms_heatmaps_out')
        self.embeddings = self.get_value_from_config('embeddings_out')
        if isinstance(linear_sum_assignment, UnsupportedPackage):
            linear_sum_assignment.raise_error(self.__provider__)
        self.decoder = AssociativeEmbeddingDecoder(
            num_joints=17,
            adjust=True,
            refine=True,
            dist_reweight=True,
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


class Pose:
    def __init__(self, num_joints, tag_size=1):
        self.num_joints = num_joints
        self.tag_size = tag_size
        self.pose = np.zeros((num_joints, 2 + 1 + tag_size), dtype=np.float32)
        self.pose_tag = np.zeros(tag_size, dtype=np.float32)
        self.valid_points_num = 0
        self.c = np.zeros(2, dtype=np.float32)

    def add(self, idx, joint, tag):
        self.pose[idx] = joint
        self.c = self.c * self.valid_points_num + joint[:2]
        self.pose_tag = (self.pose_tag * self.valid_points_num) + tag
        self.valid_points_num += 1
        self.c /= self.valid_points_num
        self.pose_tag /= self.valid_points_num

    @property
    def tag(self):
        if self.valid_points_num > 0:
            return self.pose_tag
        return None

    @property
    def center(self):
        if self.valid_points_num > 0:
            return self.c
        return None


class AssociativeEmbeddingDecoder:
    def __init__(self, num_joints, max_num_people, detection_threshold, use_detection_val,
                 ignore_too_much, tag_threshold,
                 adjust=True, refine=True, delta=0.0, joints_order=None,
                 dist_reweight=True):
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
        self.dist_reweight = dist_reweight
        self.delta = delta

    def match(self, tag_k, loc_k, val_k):
        return list(map(self._match_by_tag, zip(tag_k, loc_k, val_k)))

    @staticmethod
    def _max_match(scores):
        r, c = linear_sum_assignment(scores)
        tmp = np.stack((r, c), axis=1)
        return tmp

    def _match_by_tag(self, inp):
        tag_k, loc_k, val_k = inp
        embd_size = tag_k.shape[2]
        all_joints = np.concatenate((loc_k, val_k[..., None], tag_k), -1)

        poses = []
        for idx in self.joint_order:
            tags = tag_k[idx]
            joints = all_joints[idx]
            mask = joints[:, 2] > self.detection_threshold
            tags = tags[mask]
            joints = joints[mask]

            if len(poses) == 0:
                for tag, joint in zip(tags, joints):
                    pose = Pose(self.num_joints, embd_size)
                    pose.add(idx, joint, tag)
                    poses.append(pose)
                continue

            if joints.shape[0] == 0 or (self.ignore_too_much and len(poses) == self.max_num_people):
                continue

            poses_tags = np.stack([p.tag for p in poses], axis=0)
            diff = tags[:, None] - poses_tags[None, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if self.dist_reweight:
                # Reweight cost matrix to prefer nearby points among all that are close enough in a tag space.
                centers = np.stack([p.center for p in poses], axis=0)[None]
                dists = np.linalg.norm(joints[:, :2][:, None, :] - centers, ord=2, axis=2)
                close_tags_masks = diff_normed < self.tag_threshold
                min_dists = np.min(dists, axis=0, keepdims=True)
                dists /= min_dists + 1e-10
                diff_normed[close_tags_masks] *= dists[close_tags_masks]

            if self.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]
            num_added = diff.shape[0]
            num_grouped = diff.shape[1]
            if num_added > num_grouped:
                diff_normed = np.pad(diff_normed, ((0, 0), (0, num_added - num_grouped)),
                                     mode='constant', constant_values=1e10)

            pairs = self._max_match(diff_normed)
            for row, col in pairs:
                if row < num_added and col < num_grouped and diff_saved[row][col] < self.tag_threshold:
                    poses[col].add(idx, joints[row], tags[row])
                else:
                    pose = Pose(self.num_joints, embd_size)
                    pose.add(idx, joints[row], tags[row])
                    poses.append(pose)

        ans = np.asarray([p.pose for p in poses], dtype=np.float32).reshape(-1, self.num_joints, 2 + 1 + embd_size)
        tags = np.asarray([p.tag for p in poses], dtype=np.float32).reshape(-1, embd_size)
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

    @staticmethod
    def adjust(ans, heatmaps):
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

    @staticmethod
    def refine(heatmap, tag, keypoints, pose_tag=None):
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

        for i, (_heatmap, _tag) in enumerate(zip(heatmap, tag)):
            if keypoints[i, 2] > 0:
                continue
            # Get position with the closest tag value to the pose tag.
            diff = np.abs(_tag[..., 0] - prev_tag) + 0.5
            diff = diff.astype(np.int32).astype(_heatmap.dtype)
            diff -= _heatmap
            idx = diff.argmin()
            y, x = np.divmod(idx, _heatmap.shape[-1])
            # Corresponding keypoint detection score.
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

        np.abs(heatmaps, out=heatmaps)

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
            for i, pose in enumerate(ans):
                ans[i] = self.refine(heatmap_numpy, tag_numpy, pose, ans_tags[0][i])

        return ans, scores
