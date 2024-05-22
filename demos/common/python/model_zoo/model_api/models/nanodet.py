"""
 Copyright (c) 2022-2024 Intel Corporation

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
import math
import numpy as np

from .types import NumericalValue
from .detection_model import DetectionModel
from .utils import Detection, softmax, nms, clip_detections


class NanoDet(DetectionModel):
    __model__ = 'NanoDet'

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self._check_io_number(1, 1)
        self.output_blob_name = self._get_outputs()
        self.reg_max = 7
        self.strides = [8, 16, 32]
        self.ad = 0.5

    def _get_outputs(self):
        output_blob_name = next(iter(self.outputs))
        output_size = self.outputs[output_blob_name].shape
        if len(output_size) != 3:
            self.raise_error("Unexpected output blob shape {}. Only 3D output blob is supported".format(output_size))

        return output_blob_name

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters['resize_type'].update_default_value('fit_to_window')
        parameters['confidence_threshold'].update_default_value(0.5)
        parameters.update({
            'iou_threshold': NumericalValue(default_value=0.6, description="Threshold for NMS filtering"),
            'num_classes': NumericalValue(default_value=80, value_type=int, description="Number of classes")
        })
        return parameters

    def postprocess(self, outputs, meta):
        detections = self._parse_outputs(outputs, meta)
        detections = self.rescale_detections(detections, meta)
        return detections

    def _parse_outputs(self, outputs, meta):
        output = outputs[self.output_blob_name][0]

        cls_scores = output[:, :self.num_classes]
        bbox_preds = output[:, self.num_classes:]
        input_height, input_width = meta['padded_shape'][:2] if meta.get('padded_shape') else meta['resized_shape'][:2]

        bboxes = self.get_bboxes(bbox_preds, input_height, input_width)
        dets = []
        for label, score in enumerate(np.transpose(cls_scores)):
            mask = score > self.confidence_threshold
            filtered_boxes, score = bboxes[mask, :], score[mask]
            if score.size == 0:
                continue
            x_mins, y_mins, x_maxs, y_maxs = filtered_boxes.T
            keep = nms(x_mins, y_mins, x_maxs, y_maxs, score, self.iou_threshold, include_boundaries=True)
            score = score[keep]
            x_mins, y_mins, x_maxs, y_maxs = x_mins[keep], y_mins[keep], x_maxs[keep], y_maxs[keep]
            labels = np.full_like(score, label, dtype=int)
            dets += [Detection(*det) for det in zip(x_mins, y_mins, x_maxs, y_maxs, score, labels)]
        return dets

    @staticmethod
    def distance2bbox(points, distance, max_shape):
        x1 = np.expand_dims(points[:, 0] - distance[:, 0], -1).clip(0, max_shape[1])
        y1 = np.expand_dims(points[:, 1] - distance[:, 1], -1).clip(0, max_shape[0])
        x2 = np.expand_dims(points[:, 0] + distance[:, 2], -1).clip(0, max_shape[1])
        y2 = np.expand_dims(points[:, 1] + distance[:, 3], -1).clip(0, max_shape[0])
        return np.concatenate((x1, y1, x2, y2), axis=-1)

    def get_single_level_center_point(self, featmap_size, stride):
        h, w = featmap_size
        x_range, y_range = (np.arange(w) + self.ad) * stride, (np.arange(h) + self.ad) * stride
        y, x = np.meshgrid(y_range, x_range, indexing='ij')
        return y.flatten(), x.flatten()

    def get_bboxes(self, reg_preds, input_height, input_width):
        featmap_sizes = [(math.ceil(input_height / stride), math.ceil(input_width) / stride) for stride in self.strides]
        list_center_priors = []
        for stride, featmap_size in zip(self.strides, featmap_sizes):
            y, x = self.get_single_level_center_point(featmap_size, stride)
            strides = np.full_like(x, stride)
            list_center_priors.append(np.stack([x, y, strides, strides], axis=-1))
        center_priors = np.concatenate(list_center_priors, axis=0)
        dist_project = np.linspace(0, self.reg_max, self.reg_max + 1)
        x = np.dot(softmax(np.reshape(reg_preds, (*reg_preds.shape[:-1], 4, self.reg_max + 1)), -1, True), dist_project)
        dis_preds = x * np.expand_dims(center_priors[:, 2], -1)
        return self.distance2bbox(center_priors[:, :2], dis_preds, (input_height, input_width))

    @staticmethod
    def rescale_detections(detections, meta):
        input_h, input_w, _ = meta['resized_shape']
        orig_h, orig_w, _ = meta['original_shape']
        w = orig_w / input_w
        h = orig_h / input_h

        for detection in detections:
            detection.xmin *= w
            detection.xmax *= w
            detection.ymin *= h
            detection.ymax *= h

        return clip_detections(detections, meta['original_shape'])


class NanoDetPlus(NanoDet):
    __model__ = 'NanoDet-Plus'

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self.ad = 0
        self.strides = [8, 16, 32, 64]
