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
from scipy.optimize import linear_sum_assignment


class Detection:
    """Class that stores detected object"""

    def __init__(self, obj_id, roi, conf, waiting=0, duration=1):
        """Constructor"""

        self.id = obj_id
        self.roi = roi
        self.conf = conf
        self.waiting = waiting
        self.duration = duration

    @property
    def roi(self):
        """Returns ROI of detected object"""

        return self._roi

    @roi.setter
    def roi(self, roi):
        """Sets ROI of detected object"""

        self._roi = np.copy(roi.reshape(1, -1))


class Tracker:  # pylint: disable=too-few-public-methods
    """Class that carries out tracking of persons using Hungarian algorithm"""

    def __init__(self, detector, score_threshold, iou_threshold, smooth_weight=0.5, max_waiting=5):
        """Constructor"""

        self._detector = detector
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._smooth_weight = smooth_weight
        self._max_waiting = max_waiting

        self._last_detections = []
        self._cur_req_id, self._next_req_id = 0, 1
        self._last_id = 0

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

        union_areas = areas_set_a + areas_set_b - intersect_areas

        return intersect_areas / union_areas

    @staticmethod
    def filter_rois(new_rois, score_threshold):
        """Filters input ROIs by valid height/width and score threshold values"""

        heights = new_rois[:, 2] - new_rois[:, 0]
        widths = new_rois[:, 3] - new_rois[:, 1]
        valid_sizes_mask = np.logical_and(heights > 0.0, widths > 0.0)
        valid_conf_mask = new_rois[:, 4] > score_threshold

        valid_roi_ids = np.where(np.logical_and(valid_sizes_mask, valid_conf_mask))[0]
        filtered_rois = new_rois[valid_roi_ids, :4]
        filtered_conf = new_rois[valid_roi_ids, 4]

        return filtered_rois, filtered_conf

    def _track(self, last_detections, new_rois):
        """Updates current tracks according new observations"""

        filtered_rois, filtered_conf = self.filter_rois(new_rois, self._score_threshold)

        if filtered_rois.shape[0] == 0:
            out_detections = []
            for det in last_detections:
                det.waiting = 1
                det.duration = 0
                out_detections.append(det)

            return out_detections

        if last_detections is None or len(last_detections) == 0:
            out_detections = []
            for roi, conf in zip(filtered_rois, filtered_conf):
                out_detections.append(Detection(self._last_id, roi.reshape(1, -1), conf))
                self._last_id += 1

            return out_detections

        last_rois = np.concatenate([det.roi for det in last_detections], axis=0)
        affinity_matrix = self._matrix_iou(last_rois, filtered_rois)
        cost_matrix = 1.0 - affinity_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        affinity_values = 1.0 - cost_matrix[row_ind, col_ind]

        valid_matches = affinity_values > self._iou_threshold
        row_ind = row_ind[valid_matches]
        col_ind = col_ind[valid_matches]

        out_detections = []
        for src_id, trg_id in zip(row_ind, col_ind):
            det = last_detections[src_id]
            det.waiting = 0
            det.duration += 1
            new_roi = filtered_rois[trg_id]
            det.roi = self._smooth_roi(det.roi, new_roi.reshape(1, -1), self._smooth_weight)
            det.conf = filtered_conf[trg_id]
            out_detections.append(det)

        unmatched_src_ind = set(range(len(last_detections))) - set(row_ind.tolist())
        for src_id in unmatched_src_ind:
            det = last_detections[src_id]
            det.waiting += 1
            det.duration = 0
            if det.waiting < self._max_waiting:
                out_detections.append(det)

        unmatched_trg_ind = set(range(len(filtered_rois))) - set(col_ind.tolist())
        for trg_id in unmatched_trg_ind:
            new_roi = filtered_rois[trg_id]
            new_roi_conf = filtered_conf[trg_id]
            out_detections.append(Detection(self._last_id, new_roi.reshape(1, -1), new_roi_conf))
            self._last_id += 1

        return out_detections

    @staticmethod
    def _smooth_roi(prev_roi, new_roi, weight):
        """Smooths tracking ROI"""

        if prev_roi is None:
            return new_roi

        return weight * prev_roi + (1.0 - weight) * new_roi

    @staticmethod
    def _clip_roi(roi, frame_size):
        """Clips ROI limits according frame sizes"""

        frame_height, frame_width = frame_size

        old_roi = roi.reshape(-1)
        new_roi = [np.maximum(0, int(old_roi[0])),
                   np.maximum(0, int(old_roi[1])),
                   np.minimum(frame_width, int(old_roi[2])),
                   np.minimum(frame_height, int(old_roi[3]))]

        return np.array(new_roi)

    def _get_last_detections(self, frame_size, max_num_detections, labels_map):
        """Returns active detections"""

        if self._last_detections is None or len(self._last_detections) == 0:
            return list(), dict()

        out_detections = []
        for det in self._last_detections:
            if det.waiting > 0 or det.duration <= 1:
                continue

            clipped_roi = self._clip_roi(det.roi, frame_size)
            out_det = Detection(det.id, clipped_roi, det.conf, det.waiting, det.duration)
            out_detections.append(out_det)

        if len(out_detections) > max_num_detections:
            out_detections.sort(key=lambda x: x.conf, reverse=True)
            out_detections = out_detections[:max_num_detections]

        matched_det_ids = set(det.id for det in out_detections) & labels_map.keys()
        unused_det_ids = sorted(set(range(max_num_detections)) - matched_det_ids)

        out_labels_map = dict()
        for det in out_detections:
            if det.id in matched_det_ids:
                out_labels_map[det.id] = labels_map[det.id]
            else:
                new_local_det_id = unused_det_ids[0]
                unused_det_ids = unused_det_ids[1:]

                out_labels_map[det.id] = new_local_det_id
                det.id = new_local_det_id

        return out_detections, labels_map

    def add_frame(self, frame, max_num_detections, labels_map):
        """Adds new detections and returns active tracks"""

        self._detector.async_infer(frame, self._next_req_id)
        new_rois = self._detector.wait_request(self._cur_req_id)
        self._cur_req_id, self._next_req_id = self._next_req_id, self._cur_req_id

        if new_rois is not None:
            self._last_detections = self._track(self._last_detections, new_rois)

        frame_size = frame.shape[:2]
        out_detections, out_labels_map = self._get_last_detections(
            frame_size, max_num_detections, labels_map)

        return out_detections, out_labels_map
