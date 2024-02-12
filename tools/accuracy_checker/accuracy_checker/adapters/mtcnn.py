"""
Copyright (c) 2018-2024 Intel Corporation

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
from .adapter import Adapter
from ..config import StringField
from ..representation import DetectionPrediction


class MTCNNPAdapter(Adapter):
    __provider__ = 'mtcnn_p'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'probability_out': StringField(description='Name of Output layer with detection boxes probabilities'),
                'region_out': StringField(description='Name of output layer with detected regions'),
                'regions_format': StringField(
                    optional=True, choices=['hw', 'wh'], default='wh',
                    description='determination of coordinates order in regions, wh uses order x1y1x2y2, hw - y1x1y2x2'
                )
            }
        )

        return parameters

    def configure(self):
        self.probability_out = self.get_value_from_config('probability_out')
        self.region_out = self.get_value_from_config('region_out')
        self.regions_format = self.get_value_from_config('regions_format')
        self.outputs_verified = False

    def select_output_blob(self, outputs):
        self.probability_out = self.check_output_name(self.probability_out, outputs)
        self.region_out = self.check_output_name(self.region_out, outputs)
        self.outputs_verified = True

    @staticmethod
    def nms(boxes, threshold, overlap_type):
        """
        Args:
          boxes: [:,0:5]
          threshold: 0.5 like
          overlap_type: 'Min' or 'Union'
        Returns:
            indexes of passed boxes
        """
        if boxes.shape[0] == 0:
            return np.array([])
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
        inds = np.array(scores.argsort())

        pick = []
        while np.size(inds) > 0:
            xx1 = np.maximum(x1[inds[-1]], x1[inds[0:-1]])
            yy1 = np.maximum(y1[inds[-1]], y1[inds[0:-1]])
            xx2 = np.minimum(x2[inds[-1]], x2[inds[0:-1]])
            yy2 = np.minimum(y2[inds[-1]], y2[inds[0:-1]])
            width = np.maximum(0.0, xx2 - xx1 + 1)
            height = np.maximum(0.0, yy2 - yy1 + 1)
            inter = width * height
            if overlap_type == 'Min':
                overlap = inter / np.minimum(area[inds[-1]], area[inds[0:-1]])
            else:
                overlap = inter / (area[inds[-1]] + area[inds[0:-1]] - inter)
            pick.append(inds[-1])
            inds = inds[np.where(overlap <= threshold)[0]]

        return pick

    def process(self, raw, identifiers=None, frame_meta=None):
        if not self.outputs_verified:
            self.select_output_blob(raw)
        total_boxes_batch = self._extract_predictions(raw, frame_meta)
        results = []
        for total_boxes, identifier in zip(total_boxes_batch, identifiers):
            if np.size(total_boxes) == 0:
                results.append(DetectionPrediction(identifier, [], [], [], [], [], []))
                continue
            pick = self.nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick]
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            x_mins = total_boxes[:, 0] + total_boxes[:, 5] * regw
            y_mins = total_boxes[:, 1] + total_boxes[:, 6] * regh
            x_maxs = total_boxes[:, 2] + total_boxes[:, 7] * regw
            y_maxs = total_boxes[:, 3] + total_boxes[:, 8] * regh
            scores = total_boxes[:, 4]
            results.append(
                DetectionPrediction(identifier, np.full_like(scores, 1), scores, x_mins, y_mins, x_maxs, y_maxs)
            )

        return results

    @staticmethod
    def generate_bounding_box(mapping, reg, scale, t, r_format):
        stride = 2
        cellsize = 12
        mapping = mapping.T
        indexes = [0, 1, 2, 3] if r_format == 'wh' else [1, 0, 3, 2]
        dx1 = reg[indexes[0], :, :].T
        dy1 = reg[indexes[1], :, :].T
        dx2 = reg[indexes[2], :, :].T
        dy2 = reg[indexes[3], :, :].T
        (x, y) = np.where(mapping >= t)

        yy = y
        xx = x

        score = mapping[x, y]
        reg = np.array([dx1[x, y], dy1[x, y], dx2[x, y], dy2[x, y]])

        if reg.shape[0] == 0:
            pass
        bounding_box = np.array([yy, xx]).T

        bb1 = np.fix((stride * bounding_box + 1) / scale).T  # matlab index from 1, so with "boundingbox-1"
        bb2 = np.fix((stride * bounding_box + cellsize - 1 + 1) / scale).T  # while python don't have to
        score = np.array([score])

        bounding_box_out = np.concatenate((bb1, bb2, score, reg), axis=0)

        return bounding_box_out.T

    def _extract_predictions(self, outputs_list, meta):
        scales = [1] if not meta[0] or 'scales' not in meta[0] else meta[0]['scales']
        total_boxes = np.zeros((0, 9), float)
        for idx, outputs in enumerate(outputs_list):
            scale = scales[idx]
            mapping = outputs[self.probability_out][0, 1, :, :]
            regions = outputs[self.region_out][0]
            boxes = self.generate_bounding_box(mapping, regions, scale, 0.6, self.regions_format)
            if boxes.shape[0] != 0:
                pick = self.nms(boxes, 0.5, 'Union')

                if np.size(pick) > 0:
                    boxes = np.array(boxes)[pick, :]

            if boxes.shape[0] != 0:
                total_boxes = np.concatenate((total_boxes, boxes), axis=0)

        return [total_boxes]
