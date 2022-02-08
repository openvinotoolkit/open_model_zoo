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

import numpy as np

from ..adapters import Adapter
from ..config import StringField
from ..representation import DetectionPrediction


class PalmDetectionAdapter(Adapter):
    __provider__ = 'palm_detection'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'scores_out': StringField(description='scores output'),
            'boxes_out': StringField(description='boxes output')
        })
        return params

    def configure(self):
        self.scores_out = self.get_value_from_config('scores_out')
        self.boxes_out = self.get_value_from_config('boxes_out')
        self.outputs_verified = False

        self.num_layers = 4
        self.min_scale = 0.1484375
        self.max_scale = 0.75
        self.input_size_height = 128
        self.input_size_width = 128
        self.anchor_offset_x = 0.5
        self.anchor_offset_y = 0.5
        self.strides = [8, 16, 16, 16]
        self.aspect_ratios = [1]
        self.reduce_boxes_in_lowest_layer = False
        self.feature_map_height = []
        self.feature_map_width = []
        self.inteprolated_scale_aspect_ratio = 1
        self.fixed_anchor_size = True

        self.anchors = self.generate_anchors()

        self.sigmoid_score = True
        self.score_clipping_thresh = 100
        self.has_score_clipping_thresh = self.score_clipping_thresh != 0
        self.reverse_output_order = True
        self.keypoint_coord_offset = 4
        self.num_keypoints = 7
        self.num_values_per_keypoint = 2

        self.x_scale = 128
        self.y_scale = 128
        self.w_scale = 128
        self.h_scale = 128
        self.min_score_thresh = 0.5
        self.has_min_score_thresh = self.min_score_thresh != 0
        self.apply_exp_on_box_size = False
        self.num_classes = 1

        self.min_suppression_threshold = 0.3

    def select_output_blob(self, outputs):
        self.scores_out = self.check_output_name(self.scores_out, outputs)
        self.boxes_out = self.check_output_name(self.boxes_out, outputs)
        self.outputs_verified = True

    def process(self, raw, identifiers, frame_meta):
        result = []
        raw_output = self._extract_predictions(raw, frame_meta)
        if not self.outputs_verified:
            self.select_output_blob(raw_output)

        for identifier, raw_scores, raw_boxes in zip(identifiers, raw_output[self.scores_out],
                                                     raw_output[self.boxes_out]):
            num_boxes, _ = raw_boxes.shape
            boxes = self.decode_boxes(raw_boxes)
            detection_scores = np.zeros(num_boxes)
            detection_classes = np.zeros(num_boxes)

            for i in range(num_boxes):
                class_id = -1
                max_score = -np.inf
                for score_idx in range(self.num_classes):
                    score = raw_scores[i, score_idx]
                    if self.sigmoid_score:
                        if self.has_score_clipping_thresh:
                            score = -self.score_clipping_thresh if score < -self.score_clipping_thresh else score
                            score = self.score_clipping_thresh if score > self.score_clipping_thresh else score
                        score = 1 / (1 + np.exp(-score))
                    if max_score < score:
                        max_score = score
                        class_id = score_idx
                detection_classes[i] = class_id
                detection_scores[i] = max_score
            cond = detection_scores >= self.min_score_thresh
            boxes = np.array(boxes)[cond]
            detection_classes = detection_classes[cond]
            detection_scores = detection_scores[cond]

            cond = ((boxes[:, 2] - boxes[:, 0]) >= 0) & ((boxes[:, 3] - boxes[:, 1]) >= 0)

            boxes = boxes[cond, :]
            detection_classes = detection_classes[cond]
            detection_scores = detection_scores[cond]

            num_boxes, _ = boxes.shape

            x_mins = []
            y_mins = []
            x_maxs = []
            y_maxs = []
            labels = []
            det_scores = []
            for i in range(num_boxes):
                x_mins.append(boxes[i, 1])
                y_mins.append(boxes[i, 0])
                x_maxs.append(boxes[i, 3])
                y_maxs.append(boxes[i, 2])
                labels.append(detection_classes[i])
                det_scores.append(detection_scores[i])

            result.append(DetectionPrediction(identifier, labels, det_scores, x_mins, y_mins, x_maxs, y_maxs))

        return result

    @staticmethod
    def calculate_scale(min_scale, max_scale, stride_index, num_strides):
        return (min_scale +
                max_scale) * 0.5 if num_strides == 1 else min_scale + (max_scale -
                                                                       min_scale) * stride_index / (num_strides - 1)

    def generate_anchors(self):
        anchors = []
        layer_id = 0
        while layer_id < self.num_layers:
            anchor_height = []
            anchor_width = []
            aspect_ratios = []
            scales = []

            last_same_stride_layer = layer_id
            while last_same_stride_layer < len(self.strides) and (self.strides[last_same_stride_layer] ==
                                                                  self.strides[layer_id]):
                scale = self.calculate_scale(self.min_scale, self.max_scale, last_same_stride_layer, len(self.strides))
                ar_and_s = zip([1, 2, 0.5], [0.1, scale, scale]) if (
                    last_same_stride_layer == 0) and self.reduce_boxes_in_lowest_layer else zip(
                    self.aspect_ratios, [scale for t in self.aspect_ratios])
                for aspect_ratio, scale_ in ar_and_s:
                    aspect_ratios.append(aspect_ratio)
                    scales.append(scale_)

                if self.inteprolated_scale_aspect_ratio > 0:
                    scale_next = 1 if last_same_stride_layer == len(self.strides) - 1 else self.calculate_scale(
                        self.min_scale, self.max_scale, last_same_stride_layer + 1, len(self.strides))
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(self.inteprolated_scale_aspect_ratio)
                last_same_stride_layer += 1

            for aspect_ratio, scale in zip(aspect_ratios, scales):
                anchor_height.append(scale / np.sqrt(aspect_ratio))
                anchor_width.append(scale * np.sqrt(aspect_ratio))

            feature_map_height = self.feature_map_height[layer_id] if len(self.feature_map_height) else int(
                np.ceil(self.input_size_height / self.strides[layer_id]))
            feature_map_width = self.feature_map_width[layer_id] if len(self.feature_map_height) else int(
                np.ceil(self.input_size_width / self.strides[layer_id]))

            for y in range(feature_map_height):
                for x in range(feature_map_width):
                    for anchor_w, anchor_h in zip(anchor_width, anchor_height):
                        anchor = [(x + self.anchor_offset_x) / feature_map_width,
                                  (y + self.anchor_offset_y) / feature_map_height,
                                  1 if self.fixed_anchor_size else anchor_w,
                                  1 if self.fixed_anchor_size else anchor_h]

                        anchors.append(anchor)

            layer_id = last_same_stride_layer

        return np.array(anchors)

    def decode_boxes(self, raw_boxes):
        boxes = []
        num_boxes, _ = raw_boxes.shape

        for i in range(num_boxes):
            anchor = self.anchors[i, :]
            y_center = raw_boxes[i, 1] if self.reverse_output_order else raw_boxes[i, 0]
            x_center = raw_boxes[i, 0] if self.reverse_output_order else raw_boxes[i, 1]
            h = raw_boxes[i, 3] if self.reverse_output_order else raw_boxes[i, 2]
            w = raw_boxes[i, 2] if self.reverse_output_order else raw_boxes[i, 3]

            x_center = x_center / self.x_scale * anchor[2] + anchor[0]
            y_center = y_center / self.y_scale * anchor[3] + anchor[1]
            h = np.exp(h / self.h_scale) * anchor[3] if self.apply_exp_on_box_size else h / self.h_scale * anchor[3]
            w = np.exp(w / self.w_scale) * anchor[2] if self.apply_exp_on_box_size else w / self.w_scale * anchor[2]

            decoded = [y_center - h / 2, x_center - w / 2, y_center + h / 2, x_center + w / 2]

            for k in range(self.num_keypoints):
                offset = self.keypoint_coord_offset + k * self.num_values_per_keypoint
                keypoint_y = raw_boxes[i, offset + 1] if self.reverse_output_order else raw_boxes[i, offset]
                keypoint_x = raw_boxes[i, offset] if self.reverse_output_order else raw_boxes[i, offset + 1]

                decoded.append(keypoint_x / self.x_scale * anchor[2] + anchor[0])
                decoded.append(keypoint_y / self.y_scale * anchor[3] + anchor[1])

            boxes.append(decoded)

        return boxes
