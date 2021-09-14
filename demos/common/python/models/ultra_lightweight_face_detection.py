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
    def __init__(self, ie, model_path, resize_type='standard',
                 labels=None, threshold=0.5, iou_threshold=0.5):
        if not resize_type:
            resize_type = 'standard'
        super().__init__(ie, model_path, resize_type=resize_type,
                         labels=labels, threshold=threshold, iou_threshold=iou_threshold)
        self._check_io_number(1, 2)
        self.labels = ['Face']
        self.bboxes_blob_name, self.scores_blob_name = self._get_outputs()

    def _get_outputs(self):
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
        if self.net.outputs[bboxes_blob_name].shape[1] != self.net.outputs[scores_blob_name].shape[1]:
            raise RuntimeError("Expected the same second dimension for boxes and scores, but got {} and {}"
                               .format(self.net.outputs[bboxes_blob_name].shape,
                                       self.net.outputs[scores_blob_name].shape))
        return bboxes_blob_name, scores_blob_name

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
