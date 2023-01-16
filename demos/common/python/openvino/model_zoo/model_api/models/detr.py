"""
 Copyright (c) 2021-2023 Intel Corporation

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

from .detection_model import DetectionModel
from .utils import Detection, softmax


class DETR(DetectionModel):
    __model__ = 'DETR'

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self._check_io_number(1, 2)
        self.bboxes_blob_name, self.scores_blob_name = self._get_outputs()

    def _get_outputs(self):
        (bboxes_blob_name, bboxes_layer), (scores_blob_name, scores_layer) = self.outputs.items()

        if bboxes_layer.shape[1] != scores_layer.shape[1]:
            self.raise_error("Expected the same second dimension for boxes and scores, but got {} and {}".format(
                bboxes_layer.shape, scores_layer.shape))

        if bboxes_layer.shape[2] == 4:
            return bboxes_blob_name, scores_blob_name
        elif scores_layer.shape[2] == 4:
            return scores_blob_name, bboxes_blob_name
        else:
            self.raise_error("Expected shape [:,:,4] for bboxes output, but got {} and {}".format(
                bboxes_layer.shape, scores_layer.shape))

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters['resize_type'].update_default_value('standard')
        parameters['confidence_threshold'].update_default_value(0.5)
        return parameters

    def postprocess(self, outputs, meta):
        detections = self._parse_outputs(outputs)
        detections = self._resize_detections(detections, meta)
        return detections

    def _parse_outputs(self, outputs):
        boxes = outputs[self.bboxes_blob_name][0]
        scores = outputs[self.scores_blob_name][0]

        x_mins, y_mins, x_maxs, y_maxs = self.box_cxcywh_to_xyxy(boxes)

        scores = np.array([softmax(logit) for logit in scores])
        labels = np.argmax(scores[:, :-1], axis=-1)
        det_scores = np.max(scores[:, :-1], axis=-1)

        keep = det_scores > self.confidence_threshold

        detections = [Detection(*det) for det in zip(x_mins[keep], y_mins[keep], x_maxs[keep], y_maxs[keep],
                                                     det_scores[keep], labels[keep])]
        return detections

    @staticmethod
    def box_cxcywh_to_xyxy(box):
        x_c, y_c, w, h = box.T
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return b
