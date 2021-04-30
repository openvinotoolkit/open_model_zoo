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

import numpy as np

from ..adapters import Adapter
from ..config import ConfigValidator, StringField, ConfigError
from ..representation import PoseEstimationPrediction
from ..utils import contains_all

from .pose_estimation_associative_embedding import AssociativeEmbeddingDecoder


class HumanPoseHRNetAdapter(Adapter):
    __provider__ = 'human_pose_estimation_hrnet'
    prediction_types = (PoseEstimationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'embeddings_out': StringField(
                description="Name of output layer with associative embeddings.",
                optional=True
            ),
            'heatmaps_out': StringField(
                description="Name of output layer with keypoints heatmaps.",
                optional=True
            ),
            'nms_heatmaps_out': StringField(
                description="Name of output layer with keypoints heatmaps after NMS.",
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
        self.embeddings = self.get_value_from_config('embeddings_out')
        self.heatmaps = self.get_value_from_config('heatmaps_out')
        self.nms_heatmaps = self.get_value_from_config('nms_heatmaps_out')

        self.num_joints = 17
        self.decoder = AssociativeEmbeddingDecoder(
            num_joints=17,
            adjust=True,
            refine=True,
            dist_reweight=True,
            delta=0.5,
            max_num_people=30,
            detection_threshold=0.1,
            tag_threshold=1.0,
            use_detection_val=True,
            ignore_too_much=False)

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if not contains_all(raw_outputs, (self.embeddings, self.heatmaps, self.nms_heatmaps)):
            raise ConfigError('Some of the outputs are not found')

        raw_output = zip(identifiers, raw_outputs[self.heatmaps][None],
                         raw_outputs[self.nms_heatmaps][None],
                         raw_outputs[self.embeddings][None], frame_meta)

        for identifier, heatmap, nms_heatmap, embedding, meta in raw_output:
            h, w, _ = meta['image_size']

            # using decoder
            poses, scores = self.decoder(heatmap, embedding, nms_heatmaps=nms_heatmap)

            poses = self.transform_preds(poses, (h, w), heatmap.shape[2:])

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
