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

from .model import Model
from .utils import Detection, resize_image, nms


class FaceBoxes(Model):
    def __init__(self, ie, model_path, input_transform, threshold=0.5):
        super().__init__(ie, model_path, input_transform)

        assert len(self.net.input_info) == 1, "Expected 1 input blob"
        self.image_blob_name = next(iter(self.net.input_info))

        self._output_layer_names = sorted(self.net.outputs)
        assert len(self.net.outputs) == 2, "Expected 2 output blobs"
        self.bboxes_blob_name, self.scores_blob_name = self._parse_outputs()

        self.labels = ['Face']

        self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
        assert self.c == 3, "Expected 3-channel input"

        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.variance = [0.1, 0.2]
        self.confidence_threshold = threshold
        self.nms_threshold = 0.3
        self.keep_top_k = 750

    def _parse_outputs(self):
        bboxes_blob_name = None
        scores_blob_name = None
        for name, layer in self.net.outputs.items():
            if layer.shape[2] == 4:
                bboxes_blob_name = name
            elif layer.shape[2] == 2:
                scores_blob_name = name
            else:
                raise RuntimeError("Expected shapes [:,:,4] and [:,:2] for outputs, but got {} and {}"
                                   .format(*[output.shape for output in self.net.outputs]))
        assert self.net.outputs[bboxes_blob_name].shape[1] == self.net.outputs[scores_blob_name].shape[1], \
            "Expected the same dimension for boxes and scores"
        return bboxes_blob_name, scores_blob_name

    def preprocess(self, inputs):
        image = inputs

        resized_image = resize_image(image, (self.w, self.h))
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        resized_image = self.input_transform(resized_image)
        resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))

        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        boxes = outputs[self.bboxes_blob_name][0]
        scores = outputs[self.scores_blob_name][0]

        detections = []

        feature_maps = [[math.ceil(self.h / step), math.ceil(self.w / step)] for step in
                        self.steps]
        prior_data = self.prior_boxes(feature_maps, [self.h, self.w])

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

            keep = nms(x_mins, y_mins, x_maxs, y_maxs, filtered_score, self.nms_threshold,
                       keep_top_k=self.keep_top_k)

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

            detections = [Detection(*det, 0) for det in zip(x_mins, y_mins, x_maxs, y_maxs, filtered_score)]

        detections = self.resize_boxes(detections, meta['original_shape'][:2])
        return detections

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
    def resize_boxes(detections, image_size):
        h, w = image_size
        for detection in detections:
            detection.xmin *= w
            detection.xmax *= w
            detection.ymin *= h
            detection.ymax *= h
        return detections
