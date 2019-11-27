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
                'hm': StringField(description="Object center points heatmap."),
                'wh': StringField(description='Width heatmap'),
                'reg': StringField(description='Regression output')
            }
        )
        return parameters

    def configure(self):
        self.hm = self.get_value_from_config('hm')
        self.wh = self.get_value_from_config('wh')
        self.reg = self.get_value_from_config('reg')

    @staticmethod
    def _gather_feat(feat, ind):
        dim = feat.shape[2]
        ind = np.expand_dims(ind, axis=2)
        ind = np.repeat(ind, dim, axis=2)
        feat = feat[np.arange(feat.shape[0])[:, None, None], ind, np.arange(feat.shape[2])]
        return feat

    @staticmethod
    def _tranpose_and_gather_feat(feat, ind):
        feat = np.transpose(feat, (0, 2, 3, 1))
        feat = feat.reshape((feat.shape[0], -1, feat.shape[3]))
        feat = CTDETAdapter._gather_feat(feat, ind)
        return feat

    @staticmethod
    def _topk(scores, K=40):
        batch, cat, height, width = scores.shape

        scores = scores.reshape((batch, cat, -1))
        topk_inds = np.argpartition(scores, -K, axis=2)[:, :, -K:]
        topk_scores = scores[np.arange(scores.shape[0])[:, None, None], np.arange(scores.shape[1])[:, None], topk_inds]

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).astype(np.int32).astype(np.float)
        topk_xs = (topk_inds % width).astype(np.int32).astype(np.float)

        topk_scores = topk_scores.reshape((batch, -1))
        topk_ind = np.argpartition(topk_scores, -K, axis=1)[:, -K:]
        topk_score = topk_scores[np.arange(topk_scores.shape[0])[:, None], topk_ind]
        topk_clses = (topk_ind / K).astype(np.int32)
        topk_inds = CTDETAdapter._gather_feat(
            topk_inds.reshape((batch, -1, 1)), topk_ind).reshape((batch, K))
        topk_ys = CTDETAdapter._gather_feat(topk_ys.reshape((batch, -1, 1)), topk_ind).reshape((batch, K))
        topk_xs = CTDETAdapter._gather_feat(topk_xs.reshape((batch, -1, 1)), topk_ind).reshape((batch, K))

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

        hmax = np.array([max_pool2d(channel, kernel, pad) for channel in heat[0]])
        hmax = np.expand_dims(hmax, axis=0)
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
        for i in range(dets.shape[0]):
            dets[i, :, :2] = CTDETAdapter._transform_preds(
                dets[i, :, 0:2], center[i], scale[i], (width, heigth))
            dets[i, :, 2:4] = CTDETAdapter._transform_preds(
                dets[i, :, 2:4], center[i], scale[i], (width, heigth))
        return dets

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        hm = self._extract_predictions(raw, frame_meta)[self.hm]
        wh = self._extract_predictions(raw, frame_meta)[self.wh]
        reg = self._extract_predictions(raw, frame_meta)[self.reg]
        heat = 1/(1 + np.exp(-hm))
        batch, _, height, width = heat.shape

        heat = self._nms(heat)
        scores, inds, clses, ys, xs = self._topk(heat, K=100)
        reg = self._tranpose_and_gather_feat(reg, inds)
        K = 100

        reg = reg.reshape((batch, K, 2))
        xs = xs.reshape((batch, K, 1)) + reg[:, :, 0:1]
        ys = ys.reshape((batch, K, 1)) + reg[:, :, 1:2]

        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.reshape((batch, K, 2))
        clses = clses.reshape((batch, K, 1)).astype(np.float)
        scores = scores.reshape((batch, K, 1))
        bboxes = np.concatenate((xs - wh[..., 0:1] / 2,
                                 ys - wh[..., 1:2] / 2,
                                 xs + wh[..., 0:1] / 2,
                                 ys + wh[..., 1:2] / 2), axis=2)
        detections = np.concatenate((bboxes, scores, clses), axis=2)
        detections = np.array(detections)
        im_size = frame_meta[0].get('image_size')
        scale = max(im_size)
        center = np.array(im_size[:2])/2.0
        dets = self._transform(detections, [np.flip(center)], [scale], height, width)
        x_min, y_min, x_max, y_max, scores, classes = dets[0].transpose(1, 0)
        result.append(DetectionPrediction(identifiers, classes, scores, x_min, y_min, x_max, y_max))
        return result
