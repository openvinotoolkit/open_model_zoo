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

import random
from copy import deepcopy as copy
from collections import namedtuple

import cv2

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine, cdist

from utils.analyzer import Analyzer
from utils.misc import AverageEstimator

THE_BIGGEST_DISTANCE = 10.

TrackedObj = namedtuple('TrackedObj', 'rect label')


class ClusterFeature:
    def __init__(self, feature_len, initial_feature=None):
        self.clusters = []
        self.clusters_sizes = []
        self.feature_len = feature_len
        if initial_feature is not None:
            self.clusters.append(initial_feature)
            self.clusters_sizes.append(1)

    def update(self, feature_vec):
        if len(self.clusters) < self.feature_len:
            self.clusters.append(feature_vec)
            self.clusters_sizes.append(1)
        elif sum(self.clusters_sizes) < 2*self.feature_len:
            idx = random.randint(0, self.feature_len - 1)
            self.clusters_sizes[idx] += 1
            self.clusters[idx] += (feature_vec - self.clusters[idx]) / \
                                            self.clusters_sizes[idx]
        else:
            distances = cdist(feature_vec.reshape(1, -1),
                              np.array(self.clusters).reshape(len(self.clusters), -1), 'cosine')
            nearest_idx = np.argmin(distances)
            self.clusters_sizes[nearest_idx] += 1
            self.clusters[nearest_idx] += (feature_vec - self.clusters[nearest_idx]) / \
                                            self.clusters_sizes[nearest_idx]

    def merge(self, features, other, other_features):
        if len(features) > len(other_features):
            for feature in other_features:
                if feature is not None:
                    self.update(feature)
        else:
            for feature in features:
                if feature is not None:
                    other.update(feature)
            self.clusters = copy(other.clusters)
            self.clusters_sizes = copy(other.clusters_sizes)

    def get_clusters_matrix(self):
        return np.array(self.clusters).reshape(len(self.clusters), -1)

    def __len__(self):
        return len(self.clusters)


class OrientationFeature:
    def __init__(self, feature_len, initial_feature=(None, None)):
        assert feature_len > 0
        self.orientation_features = [AverageEstimator() for _ in range(feature_len)]
        self.is_initialized = False
        if initial_feature[0] is not None and initial_feature[1] is not None and initial_feature[1] >= 0:
            self.is_initialized = True
            self.orientation_features[initial_feature[1]].update(initial_feature[0])

    def is_valid(self):
        return self.is_initialized

    def update(self, new_feature, idx):
        if idx >= 0:
            self.is_initialized = True
            self.orientation_features[idx].update(new_feature)

    def merge(self, other):
        for f1, f2 in zip(self.orientation_features, other.orientation_features):
            f1.merge(f2)
            self.is_initialized |= f1.is_valid()

    def dist_to_other(self, other):
        distances = [1.]
        for f1, f2 in zip(self.orientation_features, other.orientation_features):
            if f1.is_valid() and f2.is_valid():
                distances.append(0.5 * cosine(f1.get(), f2.get()))
        return min(distances)

    def dist_to_vec(self, vec, orientation):
        assert orientation < len(self.orientation_features)
        if orientation >= 0 and self.orientation_features[orientation].is_valid():
            return 0.5 * cosine(vec, self.orientation_features[orientation].get())
        return 1.


def clusters_distance(clusters1, clusters2):
    if len(clusters1) > 0 and len(clusters2) > 0:
        distances = 0.5 * cdist(clusters1.get_clusters_matrix(),
                                clusters2.get_clusters_matrix(), 'cosine')
        return np.amin(distances)
    return 1.


def clusters_vec_distance(clusters, feature):
    if len(clusters) > 0 and feature is not None:
        distances = 0.5 * cdist(clusters.get_clusters_matrix(),
                                feature.reshape(1, -1), 'cosine')
        return np.amin(distances)
    return 1.


class Track:
    def __init__(self, id, cam_id, box, time, feature=None, num_clusters=4, crops=None, orientation=None):
        self.id = id
        self.cam_id = cam_id
        self.f_avg = AverageEstimator()
        self.f_clust = ClusterFeature(num_clusters)
        self.f_orient = OrientationFeature(4, (feature, orientation))
        self.features = [feature]
        self.boxes = [box]
        self.timestamps = [time]
        self.crops = [crops]
        if feature is not None:
            self.f_avg.update(feature)
            self.f_clust.update(feature)

    def get_last_feature(self):
        return self.features[-1]

    def get_end_time(self):
        return self.timestamps[-1]

    def get_start_time(self):
        return self.timestamps[0]

    def get_last_box(self):
        return self.boxes[-1]

    def __len__(self):
        return len(self.timestamps)

    def _interpolate(self, target_box, timestamp, skip_size):
        last_box = self.get_last_box()
        for t in range(1, skip_size):
            interp_box = [int(b1 + (b2 - b1) / skip_size * t) for b1, b2 in zip(last_box, target_box)]
            self.boxes.append(interp_box)
            self.timestamps.append(self.get_end_time() + 1)
            self.features.append(None)

    def _filter_last_box(self, filter_speed):
        if self.timestamps[-1] - self.timestamps[-2] == 1:
            filtered_box = list(self.boxes[-2])
            for j in range(len(self.boxes[-1])):
                filtered_box[j] = int((1 - filter_speed) * filtered_box[j]
                                      + filter_speed * self.boxes[-1][j])
            self.boxes[-1] = tuple(filtered_box)

    def add_detection(self, box, feature, timestamp, max_skip_size=1, filter_speed=0.7, crop=None):
        skip_size = timestamp - self.get_end_time()
        if 1 < skip_size <= max_skip_size:
            self._interpolate(box, timestamp, skip_size)
            assert self.get_end_time() == timestamp - 1

        self.boxes.append(box)
        self.timestamps.append(timestamp)
        self.features.append(feature)
        self._filter_last_box(filter_speed)
        if feature is not None:
            self.f_clust.update(feature)
            self.f_avg.update(feature)
        if crop is not None:
            self.crops.append(crop)

    def merge_continuation(self, other, interpolate_time_thresh=0):
        assert self.get_end_time() < other.get_start_time()
        skip_size = other.get_start_time() - self.get_end_time()
        if 1 < skip_size <= interpolate_time_thresh:
            self._interpolate(other.boxes[0], other.get_start_time(), skip_size)
            assert self.get_end_time() == other.get_start_time() - 1

        self.f_avg.merge(other.f_avg)
        self.f_clust.merge(self.features, other.f_clust, other.features)
        self.f_orient.merge(other.f_orient)
        self.timestamps += other.timestamps
        self.boxes += other.boxes
        self.features += other.features
        self.crops += other.crops


class SingleCameraTracker:
    def __init__(self, id, global_id_getter, global_id_releaser,
                 reid_model=None,
                 time_window=10,
                 continue_time_thresh=2,
                 track_clear_thresh=3000,
                 match_threshold=0.4,
                 merge_thresh=0.35,
                 n_clusters=4,
                 max_bbox_velocity=0.2,
                 detection_occlusion_thresh=0.7,
                 track_detection_iou_thresh=0.5,
                 process_curr_features_number=0,
                 visual_analyze=None,
                 interpolate_time_thresh=10,
                 detection_filter_speed=0.7,
                 rectify_thresh=0.25):
        self.reid_model = reid_model
        self.global_id_getter = global_id_getter
        self.global_id_releaser = global_id_releaser
        self.id = id
        self.tracks = []
        self.history_tracks = []
        self.time = 0
        assert time_window >= 1
        self.time_window = time_window
        assert continue_time_thresh >= 1
        self.continue_time_thresh = continue_time_thresh
        assert track_clear_thresh >= 1
        self.track_clear_thresh = track_clear_thresh
        assert 0 <= match_threshold <= 1
        self.match_threshold = match_threshold
        assert 0 <= merge_thresh <= 1
        self.merge_thresh = merge_thresh
        assert n_clusters >= 1
        self.n_clusters = n_clusters
        assert 0 <= max_bbox_velocity
        self.max_bbox_velocity = max_bbox_velocity
        assert 0 <= detection_occlusion_thresh <= 1
        self.detection_occlusion_thresh = detection_occlusion_thresh
        assert 0 <= track_detection_iou_thresh <= 1
        self.track_detection_iou_thresh = track_detection_iou_thresh
        self.process_curr_features_number = process_curr_features_number
        assert interpolate_time_thresh >= 0
        self.interpolate_time_thresh = interpolate_time_thresh
        assert 0 <= detection_filter_speed <= 1
        self.detection_filter_speed = detection_filter_speed
        self.rectify_time_thresh = self.continue_time_thresh * 4
        self.rectify_length_thresh = self.time_window // 2
        assert 0 <= rectify_thresh <= 1
        self.rectify_thresh = rectify_thresh

        self.analyzer = None
        self.current_detections = None

        if visual_analyze is not None and 'enable' in visual_analyze and visual_analyze['enable']:
            self.analyzer = Analyzer(self.id, **visual_analyze)

    def process(self, frame, detections, mask=None):
        reid_features = [None]*len(detections)
        if self.reid_model:
            reid_features = self._get_embeddings(frame, detections, mask)

        assignment = self._continue_tracks(detections, reid_features)
        self._create_new_tracks(detections, reid_features, assignment)
        self._clear_old_tracks()
        self._rectify_tracks()
        if self.time % self.time_window == 0:
            self._merge_tracks()
        if self.analyzer:
            self.analyzer.plot_timeline(self.id, self.time, self.tracks)
        self.time += 1

    def get_tracked_objects(self):
        label = 'ID'
        objs = []
        for track in self.tracks:
            if track.get_end_time() == self.time - 1 and len(track) > self.time_window:
                objs.append(TrackedObj(track.get_last_box(),
                                       label + ' ' + str(track.id)))
            elif track.get_end_time() == self.time - 1 and len(track) <= self.time_window:
                objs.append(TrackedObj(track.get_last_box(), label + ' -1'))
        return objs

    def get_tracks(self):
        return self.tracks

    def get_archived_tracks(self):
        return self.history_tracks

    def check_and_merge(self, track_source, track_candidate):
        id_candidate = track_source.id
        idx = -1
        for i, track in enumerate(self.tracks):
            if track.boxes == track_candidate.boxes:
                idx = i
        if idx < 0:  # in this case track already has been modified, merge is invalid
            return

        collisions_found = False
        for i, hist_track in enumerate(self.history_tracks):
            if hist_track.id == id_candidate \
                and not (hist_track.get_end_time() < self.tracks[idx].get_start_time()
                         or self.tracks[idx].get_end_time() < hist_track.get_start_time()):
                collisions_found = True
                break

        for i, track in enumerate(self.tracks):
            if track is not None and track.id == id_candidate:
                collisions_found = True
                break

        if not collisions_found:
            self.tracks[idx].id = id_candidate
            self.tracks[idx].f_clust.merge(self.tracks[idx].features,
                                           track_source.f_clust, track_source.features)
            track_candidate.f_clust = copy(self.tracks[idx].f_clust)
        self.tracks = list(filter(None, self.tracks))

    def _continue_tracks(self, detections, features):
        active_tracks_idx = []
        for i, track in enumerate(self.tracks):
            if track.get_end_time() >= self.time - self.continue_time_thresh:
                active_tracks_idx.append(i)

        occluded_det_idx = []
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections):
                if i != j and self._ios(det1, det2) > self.detection_occlusion_thresh:
                    occluded_det_idx.append(i)
                    features[i] = None
                    break

        cost_matrix = self._compute_detections_assignment_cost(active_tracks_idx, detections, features)

        assignment = [None for _ in range(cost_matrix.shape[0])]
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                idx = active_tracks_idx[j]
                if cost_matrix[i, j] < self.match_threshold and \
                    self._check_velocity_constraint(self.tracks[idx].get_last_box(),
                                                    self.tracks[idx].get_end_time(),
                                                    detections[i], self.time) and \
                        self._iou(self.tracks[idx].boxes[-1], detections[i]) > self.track_detection_iou_thresh:
                    assignment[i] = j

            for i, j in enumerate(assignment):
                if j is not None:
                    idx = active_tracks_idx[j]
                    crop = self.current_detections[i] if self.current_detections is not None else None
                    self.tracks[idx].add_detection(detections[i], features[i],
                                                   self.time, self.continue_time_thresh,
                                                   self.detection_filter_speed, crop)
        return assignment

    def _clear_old_tracks(self):
        clear_tracks = []
        for track in self.tracks:
            # remove too old tracks
            if track.get_end_time() < self.time - self.track_clear_thresh:
                track.features = []
                self.history_tracks.append(track)
                continue
            # remove too short and outdated tracks
            if track.get_end_time() < self.time - self.continue_time_thresh \
                    and len(track) < self.time_window:
                self.global_id_releaser(track.id)
                continue
            clear_tracks.append(track)
        self.tracks = clear_tracks

    def _rectify_tracks(self):
        active_tracks_idx = []
        not_active_tracks_idx = []
        for i, track in enumerate(self.tracks):
            if track.get_end_time() >= self.time - self.rectify_time_thresh \
                    and len(track) >= self.rectify_length_thresh:
                active_tracks_idx.append(i)
            elif len(track) >= self.rectify_length_thresh:
                not_active_tracks_idx.append(i)

        distance_matrix = np.zeros((len(active_tracks_idx),
                                    len(not_active_tracks_idx)), dtype=np.float32)
        for i, idx1 in enumerate(active_tracks_idx):
            for j, idx2 in enumerate(not_active_tracks_idx):
                distance_matrix[i, j] = self._get_rectification_distance(self.tracks[idx1], self.tracks[idx2])

        indices_rows = np.arange(distance_matrix.shape[0])
        indices_cols = np.arange(distance_matrix.shape[1])

        while len(indices_rows) > 0 and len(indices_cols) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.rectify_thresh:
                self._concatenate_tracks(active_tracks_idx[indices_rows[i]],
                                         not_active_tracks_idx[indices_cols[j]])
                distance_matrix = np.delete(distance_matrix, i, 0)
                indices_rows = np.delete(indices_rows, i)
                distance_matrix = np.delete(distance_matrix, j, 1)
                indices_cols = np.delete(indices_cols, j)
            else:
                break
        self.tracks = list(filter(None, self.tracks))

    def _get_rectification_distance(self, track1, track2):
        if (track1.get_start_time() > track2.get_end_time()
            or track2.get_start_time() > track1.get_end_time()) \
                and track1.f_avg.is_valid() and track2.f_avg.is_valid() \
                and self._check_tracks_velocity_constraint(track1, track2):
            return clusters_distance(track1.f_clust, track2.f_clust)
        return THE_BIGGEST_DISTANCE

    def _merge_tracks(self):
        distance_matrix = self._get_merge_distance_matrix()

        tracks_indices = np.arange(distance_matrix.shape[0])

        while len(tracks_indices) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.merge_thresh:
                kept_idx = self._concatenate_tracks(tracks_indices[i], tracks_indices[j])
                deleted_idx = tracks_indices[i] if kept_idx == tracks_indices[j] else tracks_indices[j]
                assert self.tracks[deleted_idx] is None
                if deleted_idx == tracks_indices[i]:
                    idx_to_delete = i
                    idx_to_update = j
                else:
                    assert deleted_idx == tracks_indices[j]
                    idx_to_delete = j
                    idx_to_update = i
                updated_row = self._get_updated_merge_distance_matrix_row(kept_idx,
                                                                          deleted_idx,
                                                                          tracks_indices)
                distance_matrix[idx_to_update, :] = updated_row
                distance_matrix[:, idx_to_update] = updated_row
                distance_matrix = np.delete(distance_matrix, idx_to_delete, 0)
                distance_matrix = np.delete(distance_matrix, idx_to_delete, 1)
                tracks_indices = np.delete(tracks_indices, idx_to_delete)
            else:
                break

        self.tracks = list(filter(None, self.tracks))

    def _get_merge_distance(self, track1, track2):
        if (track1.get_start_time() > track2.get_end_time()
            or track2.get_start_time() > track1.get_end_time()) \
                and track1.f_avg.is_valid() and track2.f_avg.is_valid() \
                and self._check_tracks_velocity_constraint(track1, track2):
            f_avg_dist = 0.5 * cosine(track1.f_avg.get(), track2.f_avg.get())
            if track1.f_orient.is_valid():
                f_complex_dist = track1.f_orient.dist_to_other(track2.f_orient)
            else:
                f_complex_dist = clusters_distance(track1.f_clust, track2.f_clust)
            return min(f_avg_dist, f_complex_dist)

        return THE_BIGGEST_DISTANCE

    def _get_merge_distance_matrix(self):
        distance_matrix = THE_BIGGEST_DISTANCE*np.eye(len(self.tracks), dtype=np.float32)
        for i, track1 in enumerate(self.tracks):
            for j, track2 in enumerate(self.tracks):
                if i < j:
                    distance_matrix[i, j] = self._get_merge_distance(track1, track2)
        distance_matrix += np.transpose(distance_matrix)
        return distance_matrix

    def _get_updated_merge_distance_matrix_row(self, update_idx, ignore_idx, alive_indices):
        distance_matrix = THE_BIGGEST_DISTANCE*np.ones(len(alive_indices), dtype=np.float32)
        for i, idx in enumerate(alive_indices):
            if idx != update_idx and idx != ignore_idx:
                distance_matrix[i] = self._get_merge_distance(self.tracks[update_idx], self.tracks[idx])
        return distance_matrix

    def _concatenate_tracks(self, i, idx):
        if self.tracks[i].get_end_time() < self.tracks[idx].get_start_time():
            self.tracks[i].merge_continuation(self.tracks[idx], self.interpolate_time_thresh)
            self.tracks[idx] = None
            return i
        else:
            assert self.tracks[idx].get_end_time() < self.tracks[i].get_start_time()
            self.tracks[idx].merge_continuation(self.tracks[i], self.interpolate_time_thresh)
            self.tracks[i] = None
            return idx

    def _create_new_tracks(self, detections, features, assignment):
        assert len(detections) == len(features)
        for i, j in enumerate(assignment):
            if j is None:
                crop = self.current_detections[i] if self.analyzer else None
                self.tracks.append(Track(self.global_id_getter(), self.id,
                                         detections[i], self.time, features[i],
                                         self.n_clusters, crop, None))

    def _compute_detections_assignment_cost(self, active_tracks_idx, detections, features):
        cost_matrix = np.zeros((len(detections), len(active_tracks_idx)), dtype=np.float32)
        if self.analyzer and len(self.tracks) > 0:
            self.analyzer.prepare_distances(self.tracks, self.current_detections)

        for i, idx in enumerate(active_tracks_idx):
            track_box = self.tracks[idx].get_last_box()
            for j, d in enumerate(detections):
                iou_dist = 0.5 * (1 - self._giou(d, track_box))
                reid_dist_curr, reid_dist_avg, reid_dist_clust = None, None, None
                if self.tracks[idx].f_avg.is_valid() and features[j] is not None \
                        and self.tracks[idx].get_last_feature() is not None:
                    reid_dist_avg = 0.5 * cosine(self.tracks[idx].f_avg.get(), features[j])
                    reid_dist_curr = 0.5 * cosine(self.tracks[idx].get_last_feature(), features[j])

                    if self.process_curr_features_number > 0:
                        num_features = len(self.tracks[idx])
                        step = -(-num_features // self.process_curr_features_number)
                        step = step if step > 0 else 1
                        start_index = 0 if self.process_curr_features_number > 1 else num_features - 1
                        for s in range(start_index, num_features - 1, step):
                            if self.tracks[idx].features[s] is not None:
                                reid_dist_curr = min(reid_dist_curr, 0.5 * cosine(self.tracks[idx].features[s], features[j]))

                    reid_dist_clust = clusters_vec_distance(self.tracks[idx].f_clust, features[j])
                    reid_dist = min(reid_dist_avg, reid_dist_curr, reid_dist_clust)
                else:
                    reid_dist = 0.5
                cost_matrix[j, i] = iou_dist * reid_dist
                if self.analyzer:
                    self.analyzer.visualize_distances(idx, j, [reid_dist_curr, reid_dist_avg, reid_dist_clust, 1 - iou_dist])
        if self.analyzer:
            self.analyzer.visualize_distances(affinity_matrix=1 - cost_matrix, active_tracks_idx=active_tracks_idx)
            self.analyzer.show_all_dist_imgs(self.time, len(self.tracks))
        return cost_matrix

    @staticmethod
    def _area(box):
        return max((box[2] - box[0]), 0) * max((box[3] - box[1]), 0)

    def _giou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersection = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                   min(b1[2], b2[2]), min(b1[3], b2[3])])

        enclosing = self._area([min(b1[0], b2[0]), min(b1[1], b2[1]),
                                max(b1[2], b2[2]), max(b1[3], b2[3])])
        u = a1 + a2 - intersection
        iou = intersection / u if u > 0 else 0
        giou = iou - (enclosing - u) / enclosing if enclosing > 0 else -1
        return giou

    def _iou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersection = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                   min(b1[2], b2[2]), min(b1[3], b2[3])])

        u = a1 + a2 - intersection
        return intersection / u if u > 0 else 0

    def _ios(self, b1, b2, a1=None, a2=None):
        # intersection over self
        if a1 is None:
            a1 = self._area(b1)
        intersection = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                   min(b1[2], b2[2]), min(b1[3], b2[3])])
        return intersection / a1 if a1 > 0 else 0

    def _get_embeddings(self, frame, detections, mask=None):
        rois = []
        embeddings = []

        if self.analyzer:
            self.current_detections = []

        for i in range(len(detections)):
            rect = detections[i]
            left, top, right, bottom = rect
            crop = frame[top:bottom, left:right]
            if mask and len(mask[i]) > 0:
                crop = cv2.bitwise_and(crop, crop, mask=mask[i])
            if left != right and top != bottom:
                rois.append(crop)

            if self.analyzer:
                self.current_detections.append(cv2.resize(crop, self.analyzer.crop_size))

        if rois:
            embeddings = self.reid_model.forward(rois)
            assert len(rois) == len(embeddings)

        return embeddings

    def _check_tracks_velocity_constraint(self, track1, track2):
        if track1.get_end_time() < track2.get_start_time():
            return self._check_velocity_constraint(track1.get_last_box(), track1.get_end_time(),
                                                   track2.boxes[0], track2.get_start_time())
        else:
            return self._check_velocity_constraint(track2.get_last_box(), track2.get_end_time(),
                                                   track1.boxes[0], track1.get_start_time())

    def _check_velocity_constraint(self, detection1, det1_time, detection2, det2_time):
        dt = abs(det2_time - det1_time)
        avg_size = 0
        for det in [detection1, detection2]:
            avg_size += 0.5 * (abs(det[2] - det[0]) + abs(det[3] - det[1]))
        avg_size *= 0.5
        shifts = [abs(x - y) for x, y in zip(detection1, detection2)]
        velocity = sum(shifts) / len(shifts) / dt / avg_size
        if velocity > self.max_bbox_velocity:
            return False
        return True
