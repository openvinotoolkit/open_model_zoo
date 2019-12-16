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

import numpy as np


class PersonTracker:  # pylint: disable=too-few-public-methods
    """ Class that allows worknig with person tracking. """

    def __init__(self, detector, score_threshold, iou_threshold, smooth_weight=0.5):
        """Constructor"""

        self._detector = detector
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._smooth_weight = smooth_weight

        self._last_roi = None
        self._cur_req_id, self._next_req_id = 0, 1

    @staticmethod
    def _matrix_iou(set_a, set_b):
        """Computes IoU metric for the two sets of vectors"""

        intersect_ymin = np.maximum(set_a[:, 0].reshape([-1, 1]), set_b[:, 0].reshape([1, -1]))
        intersect_xmin = np.maximum(set_a[:, 1].reshape([-1, 1]), set_b[:, 1].reshape([1, -1]))
        intersect_ymax = np.minimum(set_a[:, 2].reshape([-1, 1]), set_b[:, 2].reshape([1, -1]))
        intersect_xmax = np.minimum(set_a[:, 3].reshape([-1, 1]), set_b[:, 3].reshape([1, -1]))

        intersect_heights = np.maximum(0.0, intersect_ymax - intersect_ymin)
        intersect_widths = np.maximum(0.0, intersect_xmax - intersect_xmin)

        intersect_areas = intersect_heights * intersect_widths
        areas_set_a = ((set_a[:, 2] - set_a[:, 0]) * (set_a[:, 3] - set_a[:, 1])).reshape([-1, 1])
        areas_set_b = ((set_b[:, 2] - set_b[:, 0]) * (set_b[:, 3] - set_b[:, 1])).reshape([1, -1])

        areas_set_a[np.less(areas_set_a, 0.0)] = 0.0
        areas_set_b[np.less(areas_set_b, 0.0)] = 0.0

        union_areas = areas_set_a + areas_set_b - intersect_areas

        iou_values = intersect_areas / union_areas
        iou_values[np.less_equal(union_areas, 0.0)] = 0.0

        return iou_values

    def _track(self, last_roi, new_detections, score_threshold, iou_threshold):
        """Adds new detections and tracks the very first bounding box"""

        valid_ids = np.where(new_detections[:, 4] > score_threshold)[0]
        if len(valid_ids) == 0:
            return None

        filtered_detections = new_detections[valid_ids, :4]

        new_roi = None
        if last_roi is not None:
            iou_values = self._matrix_iou(last_roi.reshape([1, -1]), filtered_detections)
            iou_values = iou_values.reshape([-1])

            best_match_id = np.argmax(iou_values)
            best_match_value = iou_values[best_match_id]
            if best_match_value > iou_threshold:
                new_roi = filtered_detections[best_match_id]

        if new_roi is None:
            det_heights = filtered_detections[:, 3] - filtered_detections[:, 1]
            det_widths = filtered_detections[:, 2] - filtered_detections[:, 0]
            det_squares = det_heights * det_widths

            best_det_id = np.argmax(det_squares)
            new_roi = filtered_detections[best_det_id]

        return new_roi

    @staticmethod
    def _smooth_roi(last_roi, new_roi, weight):
        """Smooths tracking ROI"""

        if last_roi is None:
            return new_roi

        return weight * last_roi + (1.0 - weight) * new_roi

    @staticmethod
    def _clip_roi(roi, frame_size):
        """Clips ROI limits according frame sizes"""

        frame_height, frame_width = frame_size
        return [np.maximum(0, int(roi[0])),
                np.maximum(0, int(roi[1])),
                np.minimum(frame_width, int(roi[2])),
                np.minimum(frame_height, int(roi[3]))]

    def get_roi(self, frame):
        """Returns ROI of tracked person"""

        self._detector.async_infer(frame, self._next_req_id)

        detections = self._detector.wait_request(self._cur_req_id)

        out_roi = None
        if detections is not None:
            new_roi = self._track(self._last_roi, detections,
                                  self._score_threshold, self._iou_threshold)
            if new_roi is not None:
                smoothed_roi = self._smooth_roi(self._last_roi, new_roi, self._smooth_weight)
                out_roi = self._clip_roi(smoothed_roi, frame.shape[:2])

        self._cur_req_id, self._next_req_id = self._next_req_id, self._cur_req_id
        self._last_roi = np.array(out_roi, dtype=np.float32) if out_roi is not None else None

        return np.array(out_roi, dtype=np.int32) if out_roi is not None else None
