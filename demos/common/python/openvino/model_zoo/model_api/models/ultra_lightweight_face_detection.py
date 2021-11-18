"""
 Copyright (c) 2021 Intel Corporation

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
from .utils import Detection, nms


class UltraLightweightFaceDetection(DetectionModel):
    def __init__(self, model_adapter, resize_type='standard',
                 labels=None, threshold=0.5, iou_threshold=0.5):
        if not resize_type:
            resize_type = 'standard'
        super().__init__(model_adapter, resize_type=resize_type,
                         labels=labels, threshold=threshold, iou_threshold=iou_threshold)
        self._check_io_number(1, 2)
        self.labels = ['Face']
        self.bboxes_blob_name, self.scores_blob_name = self._get_outputs()

    def _get_outputs(self):
        (bboxes_blob_name, bboxes_layer), (scores_blob_name, scores_layer) = self.outputs.items()

        if bboxes_layer.shape[1] != scores_layer.shape[1]:
            raise RuntimeError("Expected the same second dimension for boxes and scores, but got {} and {}"
                               .format(bboxes_layer.shape, scores_layer.shape))

        if bboxes_layer.shape[2] == 4:
            return bboxes_blob_name, scores_blob_name
        elif scores_layer.shape[2] == 4:
            return scores_blob_name, bboxes_blob_name
        else:
            raise RuntimeError("Expected shape [:,:,4] for bboxes output, but got {} and {}"
                               .format(bboxes_layer.shape, scores_layer.shape))

    def postprocess(self, outputs, meta):
        detections = self._parse_outputs(outputs, meta)
        detections = self._resize_detections(detections, meta)
        return detections

    def _parse_outputs(self, outputs, meta):
        boxes = outputs[self.bboxes_blob_name][0]
        scores = outputs[self.scores_blob_name][0]

        score = np.transpose(scores)[1]

        mask = score > self.threshold
        filtered_boxes, filtered_score = boxes[mask, :], score[mask]

        x_mins, y_mins, x_maxs, y_maxs = filtered_boxes.T

        keep = nms(x_mins, y_mins, x_maxs, y_maxs, filtered_score, self.iou_threshold)

        filtered_score = filtered_score[keep]
        return [Detection(*det, 0) for det in zip(x_mins, y_mins, x_maxs, y_maxs, filtered_score)]
