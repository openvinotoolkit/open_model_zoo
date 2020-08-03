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

import numpy as np
from numpy.lib.stride_tricks import as_strided

from ..adapters import Adapter
from ..config import StringField
from ..preprocessor import CenterNetAffineTransformation
from ..representation import DetectionPrediction


class CTDETAdapter(Adapter):
    __provider__ = 'ctdet'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'center_heatmap_out': StringField(description="Object center points heatmap."),
                'width_height_out': StringField(description='Object size output.'),
                'regression_out': StringField(description='Regression output.')
            }
        )
        return parameters

    def configure(self):
        self.center_heatmap_out = self.get_value_from_config('center_heatmap_out')
        self.width_height_out = self.get_value_from_config('width_height_out')
        self.regression_out = self.get_value_from_config('regression_out')

    @staticmethod
    def _gather_feat(feat, ind):
        dim = feat.shape[1]
        ind = np.expand_dims(ind, axis=1)
        ind = np.repeat(ind, dim, axis=1)
        feat = feat[ind, np.arange(feat.shape[1])]
        return feat

    @staticmethod
    def _tranpose_and_gather_feat(feat, ind):
        feat = np.transpose(feat, (1, 2, 0))
        feat = feat.reshape((-1, feat.shape[2]))
        feat = CTDETAdapter._gather_feat(feat, ind)
        return feat

    @staticmethod
    def _topk(scores, K=40):
        cat, height, width = scores.shape

        scores = scores.reshape((cat, -1))
        topk_inds = np.argpartition(scores, -K, axis=1)[:, -K:]
        topk_scores = scores[np.arange(scores.shape[0])[:, None], topk_inds]

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).astype(np.int32).astype(np.float)
        topk_xs = (topk_inds % width).astype(np.int32).astype(np.float)

        topk_scores = topk_scores.reshape((-1))
        topk_ind = np.argpartition(topk_scores, -K)[-K:]
        topk_score = topk_scores[topk_ind]
        topk_clses = (topk_ind / K).astype(np.int32)
        topk_inds = CTDETAdapter._gather_feat(
            topk_inds.reshape((-1, 1)), topk_ind).reshape((K))
        topk_ys = CTDETAdapter._gather_feat(topk_ys.reshape((-1, 1)), topk_ind).reshape((K))
        topk_xs = CTDETAdapter._gather_feat(topk_xs.reshape((-1, 1)), topk_ind).reshape((K))

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    @staticmethod
    def _nms(heat, kernel=3):
        def max_pool2d(A, kernel_size, padding=1, stride=1):
            A = np.pad(A, padding, mode='constant')
            output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                            (A.shape[1] - kernel_size)//stride + 1)
            kernel_size = (kernel_size, kernel_size)
            A_w = as_strided(A, shape=output_shape + kernel_size,
                             strides=(stride*A.strides[0],
                                      stride*A.strides[1]) + A.strides)
            A_w = A_w.reshape(-1, *kernel_size)

            return A_w.max(axis=(1, 2)).reshape(output_shape)

        pad = (kernel - 1) // 2

        hmax = np.array([max_pool2d(channel, kernel, pad) for channel in heat])
        keep = (hmax == heat)
        return heat * keep

    @staticmethod
    def _transform_preds(coords, center, scale, output_size):
        def affine_transform(pt, t):
            new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
            new_pt = np.dot(t, new_pt)
            return new_pt[:2]

        target_coords = np.zeros(coords.shape)
        trans = CenterNetAffineTransformation.get_affine_transform(center, scale, 0, output_size, inv=1)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
        return target_coords

    @staticmethod
    def _transform(dets, center, scale, heigth, width):
        dets[:, :2] = CTDETAdapter._transform_preds(
            dets[:, 0:2], center, scale, (width, heigth))
        dets[:, 2:4] = CTDETAdapter._transform_preds(
            dets[:, 2:4], center, scale, (width, heigth))
        return dets

    def process(self, raw, identifiers, frame_meta):
        result = []
        predictions_batch = self._extract_predictions(raw, frame_meta)
        hm_batch = predictions_batch[self.center_heatmap_out]
        wh_batch = predictions_batch[self.width_height_out]
        reg_batch = predictions_batch[self.regression_out]
        for identifier, heat, wh, reg, meta in zip(identifiers, hm_batch, wh_batch, reg_batch, frame_meta):
            heat = np.exp(heat)/(1 + np.exp(heat))
            height, width = heat.shape[1:3]

            heat = self._nms(heat)
            scores, inds, clses, ys, xs = self._topk(heat, K=100)
            reg = self._tranpose_and_gather_feat(reg, inds)
            num_predictions = 100

            reg = reg.reshape((num_predictions, 2))
            xs = xs.reshape((num_predictions, 1)) + reg[:, 0:1]
            ys = ys.reshape((num_predictions, 1)) + reg[:, 1:2]

            wh = self._tranpose_and_gather_feat(wh, inds)
            wh = wh.reshape((num_predictions, 2))
            clses = clses.reshape((num_predictions, 1)).astype(np.float)
            scores = scores.reshape((num_predictions, 1))
            bboxes = np.concatenate((xs - wh[..., 0:1] / 2,
                                     ys - wh[..., 1:2] / 2,
                                     xs + wh[..., 0:1] / 2,
                                     ys + wh[..., 1:2] / 2), axis=1)
            detections = np.concatenate((bboxes, scores, clses), axis=1)
            im_size = meta.get('image_size')
            scale = max(im_size)
            center = np.array(im_size[:2])/2.0
            dets = self._transform(detections, np.flip(center, 0), scale, height, width)
            x_min, y_min, x_max, y_max, scores, classes = dets.transpose(1, 0)
            result.append(DetectionPrediction(identifier, classes, scores, x_min, y_min, x_max, y_max))
        return result
