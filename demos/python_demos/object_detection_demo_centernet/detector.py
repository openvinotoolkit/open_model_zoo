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

import os
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided


class Detector(object):
    def __init__(self, ie, model_path, threshold=0.3, device='CPU'):
        model = ie.read_network(model_path, os.path.splitext(model_path)[0] + '.bin')

        assert len(model.input_info) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 3, "Expected 3 output blobs"

        self._input_layer_name = next(iter(model.input_info))
        self._output_layer_names = sorted(model.outputs)

        self._ie = ie
        self._exec_model = self._ie.load_network(model, device)
        self._threshold = threshold
        self.infer_time = -1
        _, channels, self.input_height, self.input_width = model.input_info[self._input_layer_name].input_data.shape
        assert channels == 3, "Expected 3-channel input"

    @staticmethod
    def get_affine_transform(center, scale, rot, output_size, inv=False):

        def get_dir(src_point, rot_rad):
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            src_result = [0, 0]
            src_result[0] = src_point[0] * cs - src_point[1] * sn
            src_result[1] = src_point[0] * sn + src_point[1] * cs
            return src_result

        def get_3rd_point(a, b):
            direct = a - b
            return b + np.array([-direct[1], direct[0]], dtype=np.float32)

        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w, dst_h = output_size

        rot_rad = np.pi * rot / 180
        src_dir = get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        dst = np.zeros((3, 2), dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :], src[1, :] = center, center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
        src[2:, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

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
        feat = Detector._gather_feat(feat, ind)
        return feat

    @staticmethod
    def _topk(scores, K=40):
        cat, _, width = scores.shape

        scores = scores.reshape((cat, -1))
        topk_inds = np.argpartition(scores, -K, axis=1)[:, -K:]
        topk_scores = scores[np.arange(scores.shape[0])[:, None], topk_inds]

        topk_ys = (topk_inds / width).astype(np.int32).astype(np.float)
        topk_xs = (topk_inds % width).astype(np.int32).astype(np.float)

        topk_scores = topk_scores.reshape((-1))
        topk_ind = np.argpartition(topk_scores, -K)[-K:]
        topk_score = topk_scores[topk_ind]
        topk_clses = topk_ind / K
        topk_inds = Detector._gather_feat(
            topk_inds.reshape((-1, 1)), topk_ind).reshape((K))
        topk_ys = Detector._gather_feat(topk_ys.reshape((-1, 1)), topk_ind).reshape((K))
        topk_xs = Detector._gather_feat(topk_xs.reshape((-1, 1)), topk_ind).reshape((K))

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
        trans = Detector.get_affine_transform(center, scale, 0, output_size, inv=True)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
        return target_coords

    @staticmethod
    def _transform(dets, center, scale, height, width):
        dets[:, :2] = Detector._transform_preds(
            dets[:, 0:2], center, scale, (width, height))
        dets[:, 2:4] = Detector._transform_preds(
            dets[:, 2:4], center, scale, (width, height))
        return dets

    def preprocess(self, image):
        height, width = image.shape[0:2]
        center = np.array([width / 2., height / 2.], dtype=np.float32)
        scale = max(height, width)

        trans_input = self.get_affine_transform(center, scale, 0, [self.input_width, self.input_height])
        resized_image = cv2.resize(image, (width, height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (self.input_width, self.input_height),
            flags=cv2.INTER_LINEAR)

        return inp_image

    def postprocess(self, raw_output, image_sizes):
        heat, reg, wh = raw_output
        heat = heat = np.exp(heat)/(1 + np.exp(heat))
        height, width = heat.shape[1:3]
        num_predictions = 100

        heat = self._nms(heat)
        scores, inds, clses, ys, xs = self._topk(heat, K=num_predictions)
        reg = self._tranpose_and_gather_feat(reg, inds)

        reg = reg.reshape((num_predictions, 2))
        xs = xs.reshape((num_predictions, 1)) + reg[:, 0:1]
        ys = ys.reshape((num_predictions, 1)) + reg[:, 1:2]

        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.reshape((num_predictions, 2))
        clses = clses.reshape((num_predictions, 1))
        scores = scores.reshape((num_predictions, 1))
        bboxes = np.concatenate((xs - wh[..., 0:1] / 2,
                                 ys - wh[..., 1:2] / 2,
                                 xs + wh[..., 0:1] / 2,
                                 ys + wh[..., 1:2] / 2), axis=1)
        detections = np.concatenate((bboxes, scores, clses), axis=1)
        mask = detections[..., 4] >= self._threshold
        filtered_detections = detections[mask]
        scale = max(image_sizes)
        center = np.array(image_sizes[:2])/2.0
        dets = self._transform(filtered_detections, np.flip(center, 0), scale, height, width)
        return dets

    def infer(self, image):
        t0 = cv2.getTickCount()
        output = self._exec_model.infer(inputs={self._input_layer_name: image})
        self.infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        return output

    def detect(self, image):
        image_sizes = image.shape[:2]
        image = self.preprocess(image)
        image = np.transpose(image, (2, 0, 1))
        output = self.infer(image)
        detections = self.postprocess([output[name][0] for name in self._output_layer_names], image_sizes)
        return detections
