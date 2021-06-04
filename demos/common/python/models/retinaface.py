"""
 Copyright (C) 2020 Intel Corporation

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

import re

import numpy as np
from itertools import product as product

from .model import Model
from .utils import DetectionWithLandmarks, Detection, resize_image, nms


class RetinaFace(Model):
    def __init__(self, ie, model_path, threshold=0.5, mask_threshold=0.5):
        super().__init__(ie, model_path)

        assert len(self.net.input_info) == 1, "Expected 1 input blob"
        expected_outputs_count = (6, 9, 12)
        assert len(self.net.outputs) in expected_outputs_count, "Expected {} or {} output blobs".format(
            ', '.join(str(count) for count in expected_outputs_count[:-1]), int(expected_outputs_count[-1]))

        self.threshold = threshold
        self.detect_masks = len(self.net.outputs) == 12
        self.process_landmarks = len(self.net.outputs) > 6
        self.mask_threshold = mask_threshold
        self.postprocessor = RetinaFacePostprocessor(detect_attributes=self.detect_masks,
                                                     process_landmarks=self.process_landmarks)

        self.labels = ['Face'] if not self.detect_masks else ['Mask', 'No mask']

        self.image_blob_name = next(iter(self.net.input_info))
        self._output_layer_names = self.net.outputs
        self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape

    def preprocess(self, inputs):
        image = inputs

        resized_image = resize_image(image, (self.w, self.h))
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))

        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        scale_x = meta['resized_shape'][1] / meta['original_shape'][1]
        scale_y = meta['resized_shape'][0] / meta['original_shape'][0]

        outputs = self.postprocessor.process_output(outputs, scale_x, scale_y, self.threshold, self.mask_threshold)
        return outputs


class RetinaFacePostprocessor:
    def __init__(self, detect_attributes=False, process_landmarks=True):
        self._detect_masks = detect_attributes
        self._process_landmarks = process_landmarks
        _ratio = (1.,)
        self._anchor_cfg = {
            32: {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio},
            16: {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio},
            8: {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio}
        }
        self._features_stride_fpn = [32, 16, 8]
        self._anchors_fpn = dict(zip(self._features_stride_fpn, self.generate_anchors_fpn(cfg=self._anchor_cfg)))
        self._num_anchors = dict(zip(
            self._features_stride_fpn, [anchors.shape[0] for anchors in self._anchors_fpn.values()]
        ))
        self.landmark_std = 0.2 if detect_attributes else 1.0
        self.nms_threshold = 0.5 if process_landmarks else 0.3

    @staticmethod
    def generate_anchors_fpn(cfg):
        def generate_anchors(base_size=16, ratios=(0.5, 1, 2), scales=2 ** np.arange(3, 6)):
            base_anchor = np.array([1, 1, base_size, base_size]) - 1
            ratio_anchors = _ratio_enum(base_anchor, ratios)
            anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
            return anchors

        def _ratio_enum(anchor, ratios):
            w, h, x_ctr, y_ctr = _generate_wh_ctrs(anchor)
            size = w * h
            size_ratios = size / ratios
            ws = np.round(np.sqrt(size_ratios))
            hs = np.round(ws * ratios)
            anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
            return anchors

        def _scale_enum(anchor, scales):
            w, h, x_ctr, y_ctr = _generate_wh_ctrs(anchor)
            ws = w * scales
            hs = h * scales
            anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
            return anchors

        def _generate_wh_ctrs(anchor):
            w = anchor[2] - anchor[0] + 1
            h = anchor[3] - anchor[1] + 1
            x_ctr = anchor[0] + 0.5 * (w - 1)
            y_ctr = anchor[1] + 0.5 * (h - 1)
            return w, h, x_ctr, y_ctr

        def _make_anchors(ws, hs, x_ctr, y_ctr):
            ws = ws[:, np.newaxis]
            hs = hs[:, np.newaxis]
            anchors = np.hstack((
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ))
            return anchors

        rpn_feat_stride = [int(k) for k in cfg]
        rpn_feat_stride.sort(reverse=True)
        anchors = []
        for stride in rpn_feat_stride:
            feature_info = cfg[stride]
            bs = feature_info['BASE_SIZE']
            __ratios = np.array(feature_info['RATIOS'])
            __scales = np.array(feature_info['SCALES'])
            anchors.append(generate_anchors(bs, __ratios, __scales))

        return anchors

    def process_output(self, raw_output, scale_x, scale_y, face_prob_threshold, mask_prob_threshold):
        bboxes_outputs = [raw_output[name][0] for name in raw_output if re.search('.bbox.', name)]
        bboxes_outputs.sort(key=lambda x: x.shape[1])

        scores_outputs = [raw_output[name][0] for name in raw_output if re.search('.cls.', name)]
        scores_outputs.sort(key=lambda x: x.shape[1])

        if self._process_landmarks:
            landmarks_outputs = [raw_output[name][0] for name in raw_output if re.search('.landmark.', name)]
            landmarks_outputs.sort(key=lambda x: x.shape[1])

        if self._detect_masks:
            type_scores_outputs = [raw_output[name][0] for name in raw_output if re.search('.type.', name)]
            type_scores_outputs.sort(key=lambda x: x.shape[1])

        proposals_list = []
        scores_list = []
        landmarks_list = []
        mask_scores_list = []
        for idx, s in enumerate(self._features_stride_fpn):
            anchor_num = self._num_anchors[s]
            scores = self._get_scores(scores_outputs[idx], anchor_num)
            bbox_deltas = bboxes_outputs[idx]
            height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]
            anchors_fpn = self._anchors_fpn[s]
            anchors = self.anchors_plane(height, width, int(s), anchors_fpn)
            anchors = anchors.reshape((height * width * anchor_num, 4))
            proposals = self._get_proposals(bbox_deltas, anchor_num, anchors)
            threshold_mask = scores >= face_prob_threshold

            proposals_list.extend(proposals[threshold_mask, :])
            scores_list.extend(scores[threshold_mask])
            if self._process_landmarks:
                landmarks = self._get_landmarks(landmarks_outputs[idx], anchor_num, anchors)
                landmarks_list.extend(landmarks[threshold_mask, :])
            if self._detect_masks:
                masks = self._get_mask_scores(type_scores_outputs[idx], anchor_num)
                mask_scores_list.extend(masks[threshold_mask])

        if len(scores_list) > 0:
            proposals_list = np.array(proposals_list)
            scores_list = np.array(scores_list)
            landmarks_list = np.array(landmarks_list)
            mask_scores_list = np.array(mask_scores_list)
            x_mins, y_mins, x_maxs, y_maxs = proposals_list.T
            keep = nms(x_mins, y_mins, x_maxs, y_maxs, scores_list, self.nms_threshold,
                       include_boundaries=not self._process_landmarks)
            proposals_list = proposals_list[keep]
            scores_list = scores_list[keep]
            if self._process_landmarks:
                landmarks_list = landmarks_list[keep]
            if self._detect_masks:
                mask_scores_list = mask_scores_list[keep]

        result = []
        if len(scores_list) != 0:
            scores = np.reshape(scores_list, -1)
            mask_scores_list = np.reshape(mask_scores_list, -1)
            x_mins, y_mins, x_maxs, y_maxs = np.array(proposals_list).T # pylint: disable=E0633
            x_mins /= scale_x
            x_maxs /= scale_x
            y_mins /= scale_y
            y_maxs /= scale_y

            result = []
            if self._process_landmarks:
                landmarks_x_coords = np.array(landmarks_list)[:, :, ::2].reshape(len(landmarks_list), -1) / scale_x
                landmarks_y_coords = np.array(landmarks_list)[:, :, 1::2].reshape(len(landmarks_list), -1) / scale_y
                if self._detect_masks:
                    for i in range(len(scores_list)):
                        result.append(DetectionWithLandmarks(x_mins[i], y_mins[i], x_maxs[i], y_maxs[i], scores[i],
                                                             0 if mask_scores_list[i] > mask_prob_threshold else 1,
                                                             landmarks_x_coords[i], landmarks_y_coords[i]))
                else:
                    for i in range(len(scores_list)):
                        result.append(DetectionWithLandmarks(x_mins[i], y_mins[i], x_maxs[i], y_maxs[i], scores[i], 0,
                                                             landmarks_x_coords[i], landmarks_y_coords[i]))
            else:
                for i in range(len(scores_list)):
                    result.append(Detection(x_mins[i], y_mins[i], x_maxs[i], y_maxs[i], scores[i], 0))

        return result

    def _get_proposals(self, bbox_deltas, anchor_num, anchors):
        bbox_deltas = bbox_deltas.transpose((1, 2, 0))
        bbox_pred_len = bbox_deltas.shape[2] // anchor_num
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        proposals = self.bbox_pred(anchors, bbox_deltas)
        return proposals

    @staticmethod
    def _get_scores(scores, anchor_num):
        scores = scores[anchor_num:, :, :]
        scores = scores.transpose((1, 2, 0)).reshape(-1)
        return scores

    @staticmethod
    def _get_mask_scores(type_scores, anchor_num):
        mask_scores = type_scores[anchor_num * 2:, :, :]
        mask_scores = mask_scores.transpose((1, 2, 0)).reshape(-1)
        return mask_scores

    def _get_landmarks(self, landmark_deltas, anchor_num, anchors):
        landmark_pred_len = landmark_deltas.shape[0] // anchor_num
        landmark_deltas = landmark_deltas.transpose((1, 2, 0)).reshape((-1, 5, landmark_pred_len // 5))
        landmark_deltas *= self.landmark_std
        landmarks = self.landmark_pred(anchors, landmark_deltas)
        return landmarks

    @staticmethod
    def bbox_pred(boxes, box_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]
        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]
        pred_boxes = np.zeros(box_deltas.shape)
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1] > 4:
            pred_boxes[:, 4:] = box_deltas[:, 4:]

        return pred_boxes

    @staticmethod
    def anchors_plane(height, width, stride, base_anchors):
        num_anchors = base_anchors.shape[0]
        all_anchors = np.zeros((height, width, num_anchors, 4))
        for iw in range(width):
            sw = iw * stride
            for ih in range(height):
                sh = ih * stride
                for k in range(num_anchors):
                    all_anchors[ih, iw, k, 0] = base_anchors[k, 0] + sw
                    all_anchors[ih, iw, k, 1] = base_anchors[k, 1] + sh
                    all_anchors[ih, iw, k, 2] = base_anchors[k, 2] + sw
                    all_anchors[ih, iw, k, 3] = base_anchors[k, 3] + sh

        return all_anchors

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
            pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y

        return pred


class RetinaFacePyTorch(Model):
    def __init__(self, ie, model_path, threshold=0.5):
        super().__init__(ie, model_path)

        assert len(self.net.input_info) == 1, "Expected 1 input blob"
        expected_outputs_count = (2, 3)
        assert len(self.net.outputs) in expected_outputs_count, "Expected {} or {} output blobs".format(
            expected_outputs_count[0], expected_outputs_count[1])

        self.threshold = threshold
        self.process_landmarks = len(self.net.outputs) == 3
        self.postprocessor = RetinaFacePyTorchPostprocessor(process_landmarks=self.process_landmarks)

        self.labels = ['Face']

        self.image_blob_name = next(iter(self.net.input_info))
        self._output_layer_names = self.net.outputs
        self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape

    def preprocess(self, inputs):
        image = inputs

        resized_image = resize_image(image, (self.w, self.h))
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        resized_image = np.expand_dims(resized_image.transpose((2, 0, 1)), axis=0) # Change data layout from HWC to CHW

        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        scale_x = meta['resized_shape'][1] / meta['original_shape'][1]
        scale_y = meta['resized_shape'][0] / meta['original_shape'][0]

        outputs = self.postprocessor.process_output(outputs, scale_x, scale_y, self.threshold,
                                                    meta['resized_shape'][:2])
        return outputs


class RetinaFacePyTorchPostprocessor:
    def __init__(self, process_landmarks=True):
        self._process_landmarks = process_landmarks
        self.nms_threshold = 0.5 if process_landmarks else 0.3
        self.variance = [0.1, 0.2]

    def process_output(self, raw_output, scale_x, scale_y, face_prob_threshold, image_size):
        bboxes_output = [raw_output[name][0] for name in raw_output if re.search('.bbox.', name)][0]

        scores_output = [raw_output[name][0] for name in raw_output if re.search('.cls.', name)][0]

        if self._process_landmarks:
            landmarks_output = [raw_output[name][0] for name in raw_output if re.search('.landmark.', name)][0]

        prior_data = self.generate_prior_data(image_size)
        proposals = self._get_proposals(bboxes_output, prior_data, image_size)
        scores = scores_output[:, 1]
        filter_idx = np.where(scores > face_prob_threshold)[0]
        proposals = proposals[filter_idx]
        scores = scores[filter_idx]
        if self._process_landmarks:
            landmarks = self._get_landmarks(landmarks_output, prior_data,
                                            image_size)
            landmarks = landmarks[filter_idx]

        if np.size(scores) > 0:

            x_mins, y_mins, x_maxs, y_maxs = proposals.T
            keep = nms(x_mins, y_mins, x_maxs, y_maxs, scores, self.nms_threshold,
                       include_boundaries=not self._process_landmarks)

            proposals = proposals[keep]
            scores = scores[keep]
            if self._process_landmarks:
                landmarks = landmarks[keep]

        result = []
        if np.size(scores) != 0:
            scores = np.reshape(scores, -1)
            x_mins, y_mins, x_maxs, y_maxs = np.array(proposals).T # pylint: disable=E0633
            x_mins /= scale_x
            x_maxs /= scale_x
            y_mins /= scale_y
            y_maxs /= scale_y

            result = []
            if self._process_landmarks:
                landmarks_x_coords = np.array(landmarks)[:, ::2] / scale_x
                landmarks_y_coords = np.array(landmarks)[:, 1::2] / scale_y
                for x_min, y_min, x_max, y_max, score, landmarks_x, landmarks_y in zip(
                    x_mins, y_mins, x_maxs, y_maxs, scores, landmarks_x_coords, landmarks_y_coords):
                    result.append(DetectionWithLandmarks(x_min, y_min, x_max, y_max, score, 0, landmarks_x,
                                                         landmarks_y))
            else:
                for x_min, y_min, x_max, y_max, score in zip(x_mins, y_mins, x_maxs, y_maxs, scores):
                    result.append(Detection(x_min, y_min, x_max, y_max, score, 0))

        return result

    @staticmethod
    def generate_prior_data(image_size):
        global_min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        anchors = []
        feature_maps = [[int(np.rint(image_size[0]/step)), int(np.rint(image_size[1]/step))] for step in steps]
        for idx, feature_map in enumerate(feature_maps):
            min_sizes = global_min_sizes[idx]
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in min_sizes:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    dense_cx = [x * steps[idx] / image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * steps[idx] / image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        priors = np.array(anchors).reshape((-1, 4))
        return priors

    def _get_proposals(self, raw_boxes, priors, image_size):
        proposals = self.decode_boxes(raw_boxes, priors, self.variance)
        proposals[:, ::2] = proposals[:, ::2] * image_size[1]
        proposals[:, 1::2] = proposals[:, 1::2] * image_size[0]
        return proposals

    @staticmethod
    def decode_boxes(raw_boxes, priors, variance):
        boxes = np.concatenate((
            priors[:, :2] + raw_boxes[:, :2] * variance[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(raw_boxes[:, 2:] * variance[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def _get_landmarks(self, raw_landmarks, priors, image_size):
        landmarks = self.decode_landmarks(raw_landmarks, priors, self.variance)
        landmarks[:, ::2] = landmarks[:, ::2] * image_size[1]
        landmarks[:, 1::2] = landmarks[:, 1::2] * image_size[0]
        return landmarks

    @staticmethod
    def decode_landmarks(raw_landmarks, priors, variance):
        landmarks = np.concatenate((priors[:, :2] + raw_landmarks[:, :2] * variance[0] * priors[:, 2:],
                                    priors[:, :2] + raw_landmarks[:, 2:4] * variance[0] * priors[:, 2:],
                                    priors[:, :2] + raw_landmarks[:, 4:6] * variance[0] * priors[:, 2:],
                                    priors[:, :2] + raw_landmarks[:, 6:8] * variance[0] * priors[:, 2:],
                                    priors[:, :2] + raw_landmarks[:, 8:10] * variance[0] * priors[:, 2:]), 1)
        return landmarks
