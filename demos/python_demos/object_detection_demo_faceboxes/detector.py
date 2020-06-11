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
import itertools
import math
import numpy as np
import os
import cv2
from collections import namedtuple

class Detector(object):
    def __init__(self, ie, model_path, device='CPU', threshold=0.5):
        model = ie.read_network(model=model_path, weights=os.path.splitext(model_path)[0] + '.bin')

        assert len(model.input_info) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 2, "Expected 2 output blobs"

        self._input_layer_name = next(iter(model.input_info))
        self._output_layer_names = sorted(model.outputs)

        assert model.outputs[self._output_layer_names[0]].shape[1] == \
               model.outputs[self._output_layer_names[1]].shape[1], "Expected the same dimension for boxes and scores"
        assert model.outputs[self._output_layer_names[0]].shape[2] == 4, "Expected 4-coordinate boxes"
        assert model.outputs[self._output_layer_names[1]].shape[2] == 2, "Expected 2-class scores(background, face)"

        self._ie = ie
        self._exec_model = self._ie.load_network(model, device)
        self.infer_time = -1
        _, channels, self.input_height, self.input_width = model.input_info[self._input_layer_name].input_data.shape
        assert channels == 3, "Expected 3-channel input"

        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.variance = [0.1, 0.2]
        self.confidence_threshold = threshold
        self.nms_threshold = 0.3
        self.keep_top_k = 750

    @staticmethod
    def calculate_anchors(list_x, list_y, min_size, image_size, step):
        anchors = []
        s_kx = min_size / image_size[1]
        s_ky = min_size / image_size[0]
        dense_cx = [x * step / image_size[1] for x in list_x]
        dense_cy = [y * step / image_size[0] for y in list_y]
        for cy, cx in itertools.product(dense_cy, dense_cx):
            anchors.append([cx, cy, s_kx, s_ky])
        return anchors

    def calculate_anchors_zero_level(self, f_x, f_y, min_sizes, image_size, step):
        anchors = []
        for min_size in min_sizes:
            if min_size == 32:
                list_x = [f_x + 0, f_x + 0.25, f_x + 0.5, f_x + 0.75]
                list_y = [f_y + 0, f_y + 0.25, f_y + 0.5, f_y + 0.75]
            elif min_size == 64:
                list_x = [f_x + 0, f_x + 0.5]
                list_y = [f_y + 0, f_y + 0.5]
            else:
                list_x = [f_x + 0.5]
                list_y = [f_y + 0.5]
            anchors.extend(self.calculate_anchors(list_x, list_y, min_size, image_size, step))
        return anchors

    def prior_boxes(self, feature_maps, image_size):
        anchors = []
        for k, f in enumerate(feature_maps):
            for i, j in itertools.product(range(f[0]), range(f[1])):
                if k == 0:
                    anchors.extend(self.calculate_anchors_zero_level(j, i, self.min_sizes[k],
                                                                     image_size, self.steps[k]))
                else:
                    anchors.extend(self.calculate_anchors([j + 0.5], [i + 0.5], self.min_sizes[k][0],
                                                          image_size, self.steps[k]))
        anchors = np.clip(anchors, 0, 1)

        return anchors

    @staticmethod
    def nms(x1, y1, x2, y2, scores, thresh, include_boundaries=True, keep_top_k=None):
        b = 1 if include_boundaries else 0

        areas = (x2 - x1 + b) * (y2 - y1 + b)
        order = scores.argsort()[::-1]

        if keep_top_k:
            order = order[:keep_top_k]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + b)
            h = np.maximum(0.0, yy2 - yy1 + b)
            intersection = w * h

            union = (areas[i] + areas[order[1:]] - intersection)
            overlap = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)

            order = order[np.where(overlap <= thresh)[0] + 1]  # pylint: disable=W0143

        return keep

    @staticmethod
    def resize_boxes(detections, image_size):
        h, w = image_size
        for i, detection in enumerate(detections):
            detections[i] = detection._replace(x_min=detection.x_min * w,
                                                   y_min=detection.y_min * h,
                                                   x_max=detection.x_max * w,
                                                   y_max=detection.y_max * h)
        return detections

    def preprocess(self, image):
        return cv2.resize(image, (self.input_width, self.input_height))

    def infer(self, image):
        t0 = cv2.getTickCount()
        inputs = {self._input_layer_name: image}
        output = self._exec_model.infer(inputs=inputs)
        self.infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        return output

    def postprocess(self, raw_output, image_sizes):
        boxes, scores = raw_output

        detection = namedtuple('detection', 'score, x_min, y_min, x_max, y_max')
        detections = []
        image_info = [self.input_height, self.input_width]

        feature_maps = [[math.ceil(image_info[0] / step), math.ceil(image_info[1] / step)] for step in
                        self.steps]
        prior_data = self.prior_boxes(feature_maps, image_info)

        boxes[:, :2] = self.variance[0] * boxes[:, :2]
        boxes[:, 2:] = self.variance[1] * boxes[:, 2:]
        boxes[:, :2] = boxes[:, :2] * prior_data[:, 2:] + prior_data[:, :2]
        boxes[:, 2:] = np.exp(boxes[:, 2:]) * prior_data[:, 2:]

        score = np.transpose(scores)[1]

        mask = score > self.confidence_threshold
        filtered_boxes, filtered_score = boxes[mask, :], score[mask]
        if filtered_score.size != 0:
            x_mins = (filtered_boxes[:, 0] - 0.5 * filtered_boxes[:, 2])
            y_mins = (filtered_boxes[:, 1] - 0.5 * filtered_boxes[:, 3])
            x_maxs = (filtered_boxes[:, 0] + 0.5 * filtered_boxes[:, 2])
            y_maxs = (filtered_boxes[:, 1] + 0.5 * filtered_boxes[:, 3])

            keep = self.nms(x_mins, y_mins, x_maxs, y_maxs, filtered_score, self.nms_threshold,
                           include_boundaries=False, keep_top_k=self.keep_top_k)

            filtered_score = filtered_score[keep]
            x_mins = x_mins[keep]
            y_mins = y_mins[keep]
            x_maxs = x_maxs[keep]
            y_maxs = y_maxs[keep]

            if filtered_score.size > self.keep_top_k:
                filtered_score = filtered_score[:self.keep_top_k]
                x_mins = x_mins[:self.keep_top_k]
                y_mins = y_mins[:self.keep_top_k]
                x_maxs = x_maxs[:self.keep_top_k]
                y_maxs = y_maxs[:self.keep_top_k]

            for score, x_min, y_min, x_max, y_max in zip(filtered_score, x_mins, y_mins, x_maxs, y_maxs):
                detections.append(detection(score=score, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))

        detections = self.resize_boxes(detections, image_sizes)
        return detections

    def detect(self, image):
        image_sizes = image.shape[:2]
        image = self.preprocess(image)
        image = np.transpose(image, (2, 0, 1))
        output = self.infer(image)
        detections = self.postprocess([output[name][0] for name in self._output_layer_names], image_sizes)
        return detections
