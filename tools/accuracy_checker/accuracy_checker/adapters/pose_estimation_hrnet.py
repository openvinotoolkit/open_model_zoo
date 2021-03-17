"""
Copyright (c) 2020-2021 Intel Corporation

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

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, ConfigError
from ..representation import PoseEstimationPrediction

from .pose_estimation_openpose import HeatmapNMS
from .pose_estimation_associative_embedding import AssociativeEmbeddingDecoder


def contains_all(container, args):
    return set(container).intersection(args) == set(args)


class HumanPoseHRNetAdapter(Adapter):
    __provider__ = 'human_pose_estimation_hrnet'
    prediction_types = (PoseEstimationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'heatmaps_lr_and_embeddings_out': StringField(
                description="Name of output layer with keypoints heatmaps ans associative embeddings.",
                optional=True
            ),
            'heatmaps_out': StringField(
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
        self.heatmaps_lr_and_tags = self.get_value_from_config('heatmaps_lr_and_embeddings_out')
        self.heatmaps = self.get_value_from_config('heatmaps_out')
        self.nms = HeatmapNMS(kernel=5)
        self.num_joints = 17
        self.decoder = AssociativeEmbeddingHRNetDecoder(
            num_joints=17,
            adjust=True,
            refine=True,
            dist_reweight=True,
            delta=0.0,
            max_num_people=30,
            detection_threshold=0.1,
            tag_threshold=1.0,
            use_detection_val=True,
            ignore_too_much=False)

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not contains_all(raw_outputs, (self.heatmaps_lr_and_tags, self.heatmaps)):
            raise ConfigError('Some of the outputs are not found')

        heatmaps_lr_and_tags = raw_outputs[self.heatmaps_lr_and_tags][None]
        heatmaps = raw_outputs[self.heatmaps][None]
        raw_output = zip(identifiers, heatmaps_lr_and_tags, heatmaps, frame_meta)

        for identifier, heatmap_lr_and_tag, heatmap, meta in raw_output:
            h, w, _ = meta['image_size']
            # resize heatmaps_lr and tags to size of input layer
            heatmap_lr_and_tag = np.transpose(heatmap_lr_and_tag[0], (1, 2, 0))
            heatmap_and_tag = cv2.resize(heatmap_lr_and_tag, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

            heatmap = np.transpose(heatmap[0], (1, 2, 0))

            heatmaps = (heatmap_and_tag[:, :, :self.num_joints] + heatmap) / 2
            heatmaps = cv2.resize(heatmaps, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            heatmaps = np.transpose(heatmaps, (2, 0, 1))[None]
            tags = heatmap_and_tag[:, :, self.num_joints:]
            tags = cv2.resize(tags, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            tags = np.transpose(tags, (2, 0, 1))[None]
            nms_heatmaps = self.nms(heatmaps)

            # using decoder
            poses, scores = self.decoder(heatmaps, tags, nms_heatmaps=nms_heatmaps)

            poses = self.transform_preds(poses, (h, w), heatmaps.shape[2:])

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

            point_scores = poses[:, :, 2]
            result.append(PoseEstimationPrediction(
                identifier,
                poses[:, :, 0],
                poses[:, :, 1],
                point_scores,
                scores))
        return result

    def transform_preds(self, poses, input_size, output_size):
        # compute trans
        scale = min(input_size) / min(output_size)
        trans = np.zeros((2, 3))
        trans[0][0] = trans[1][1] = scale
        shift = (max(input_size) - max(output_size) * scale) / 2
        trans[np.argmax((input_size[::-1]))][2] = shift

        # affine_trans
        for person in poses:
            for joint in person:
                joint[:2] = self.affine_transform(joint[:2], trans)
        return poses

    @staticmethod
    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.])
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]


class AssociativeEmbeddingHRNetDecoder(AssociativeEmbeddingDecoder):
    @staticmethod
    def adjust(ans, heatmaps):
        H, W = heatmaps.shape[-2:]
        for n, people in enumerate(ans):
            for person in people:
                for k, joint in enumerate(person):
                    if joint[2] > 0:
                        heatmap = heatmaps[n, k]
                        px = int(joint[0])
                        py = int(joint[1])

                        diff = np.array([
                            heatmap[py, min(px + 1, W - 1)] - heatmap[py, max(px - 1, 0)],
                            heatmap[min(py + 1, H - 1), px] - heatmap[max(py - 1, 0), px]
                        ])
                        joint[:2] += np.sign(diff) * .25 + 0.5
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
                diff = np.array([
                    _heatmap[y, min(x + 1, W - 1)] - _heatmap[y, max(x - 1, 0)],
                    _heatmap[min(y + 1, H - 1), x] - _heatmap[max(y - 1, 0), x]
                ])
                keypoints[i, :2] += np.sign(diff) * .25 + 0.5

        return keypoints
