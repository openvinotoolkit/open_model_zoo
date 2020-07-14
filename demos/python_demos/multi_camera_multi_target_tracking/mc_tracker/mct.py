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

import queue

import numpy as np
from scipy.spatial.distance import cosine

from .sct import SingleCameraTracker, clusters_distance, THE_BIGGEST_DISTANCE


class MultiCameraTracker:
    def __init__(self, num_sources, reid_model,
                 sct_config={},
                 time_window=20,
                 global_match_thresh=0.35,
                 bbox_min_aspect_ratio=1.2,
                 visual_analyze=None,
                 ):
        self.scts = []
        self.time = 0
        self.last_global_id = 0
        self.global_ids_queue = queue.Queue()
        assert time_window >= 1
        self.time_window = time_window  # should be greater than time window in scts
        assert 0 <= global_match_thresh <= 1
        self.global_match_thresh = global_match_thresh
        assert bbox_min_aspect_ratio >= 0
        self.bbox_min_aspect_ratio = bbox_min_aspect_ratio
        assert num_sources > 0
        for i in range(num_sources):
            self.scts.append(SingleCameraTracker(i, self._get_next_global_id,
                                                 self._release_global_id,
                                                 reid_model, visual_analyze=visual_analyze, **sct_config))

    def process(self, frames, all_detections, masks=None):
        assert len(frames) == len(all_detections) == len(self.scts)
        all_tracks = []
        for i, sct in enumerate(self.scts):
            if masks:
                mask = masks[i]
            else:
                mask = None
            if self.bbox_min_aspect_ratio is not None:
                all_detections[i], mask = self._filter_detections(all_detections[i], mask)
            sct.process(frames[i], all_detections[i], mask)
            all_tracks += sct.get_tracks()

        if self.time > 0 and self.time % self.time_window == 0:
            self._merge_all(all_tracks)

        self.time += 1

    def _merge_all(self, all_tracks):
        distance_matrix = self._compute_mct_distance_matrix(all_tracks)
        indices_rows = np.arange(distance_matrix.shape[0])
        indices_cols = np.arange(distance_matrix.shape[1])

        while len(indices_rows) > 0 and len(indices_cols) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.global_match_thresh:
                idx1, idx2 = indices_rows[i], indices_cols[j]
                if all_tracks[idx1].id > all_tracks[idx2].id:
                    self.scts[all_tracks[idx1].cam_id].check_and_merge(all_tracks[idx2], all_tracks[idx1])
                else:
                    self.scts[all_tracks[idx2].cam_id].check_and_merge(all_tracks[idx1], all_tracks[idx2])
                assert i != j
                distance_matrix = np.delete(distance_matrix, max(i, j), 0)
                distance_matrix = np.delete(distance_matrix, max(i, j), 1)
                distance_matrix = np.delete(distance_matrix, min(i, j), 0)
                distance_matrix = np.delete(distance_matrix, min(i, j), 1)
                indices_rows = np.delete(indices_rows, max(i, j))
                indices_rows = np.delete(indices_rows, min(i, j))
                indices_cols = np.delete(indices_cols, max(i, j))
                indices_cols = np.delete(indices_cols, min(i, j))
            else:
                break

    def _filter_detections(self, detections, masks):
        clean_detections = []
        clean_masks = []
        for i, det in enumerate(detections):
            w = det[2] - det[0]
            h = det[3] - det[1]
            ar = h / w
            if ar > self.bbox_min_aspect_ratio:
                clean_detections.append(det)
                if i < len(masks):
                    clean_masks.append(masks[i])
        return clean_detections, clean_masks

    def _compute_mct_distance_matrix(self, all_tracks):
        distance_matrix = THE_BIGGEST_DISTANCE * np.eye(len(all_tracks), dtype=np.float32)
        for i, track1 in enumerate(all_tracks):
            for j, track2 in enumerate(all_tracks):
                if j >= i:
                    break
                if track1.id != track2.id and track1.cam_id != track2.cam_id and \
                        len(track1) > self.time_window and len(track2) > self.time_window and \
                        track1.f_avg.is_valid() and track2.f_avg.is_valid():
                    if not track1.f_orient.is_valid():
                        f_complex_dist = clusters_distance(track1.f_clust, track2.f_clust)
                    else:
                        f_complex_dist = track1.f_orient.dist_to_other(track2.f_orient)
                    f_avg_dist = 0.5 * cosine(track1.f_avg.get(), track2.f_avg.get())
                    distance_matrix[i, j] = min(f_avg_dist, f_complex_dist)
                else:
                    distance_matrix[i, j] = THE_BIGGEST_DISTANCE
        return distance_matrix + np.transpose(distance_matrix)

    def _get_next_global_id(self):
        if self.global_ids_queue.empty():
            self.global_ids_queue.put(self.last_global_id)
            self.last_global_id += 1

        return self.global_ids_queue.get_nowait()

    def _release_global_id(self, id):
        assert id <= self.last_global_id
        self.global_ids_queue.put(id)

    def get_tracked_objects(self):
        return [sct.get_tracked_objects() for sct in self.scts]

    def get_all_tracks_history(self):
        history = []
        for sct in self.scts:
            cam_tracks = sct.get_archived_tracks() + sct.get_tracks()
            for i in range(len(cam_tracks)):
                cam_tracks[i] = {'id': cam_tracks[i].id,
                                 'timestamps':  cam_tracks[i].timestamps,
                                 'boxes': cam_tracks[i].boxes}
            history.append(cam_tracks)
        return history
