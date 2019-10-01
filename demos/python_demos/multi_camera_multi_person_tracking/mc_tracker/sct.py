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

from utils.misc import none_to_zero

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
            self.clusters[idx] += (feature_vec - self.clusters[idx]) / \
                                            self.clusters_sizes[idx]
            self.clusters_sizes[idx] += 1
        else:
            distances = cdist(feature_vec.reshape(1, -1),
                              np.array(self.clusters).reshape(len(self.clusters), -1), 'cosine')
            nearest_idx = np.argmin(distances)
            self.clusters_sizes[nearest_idx] += 1
            self.clusters[nearest_idx] += (feature_vec - self.clusters[nearest_idx]) / \
                                            self.clusters_sizes[nearest_idx]

    def get_clusters_matrix(self):
        return np.array(self.clusters).reshape(len(self.clusters), -1)

    def __len__(self):
        return len(self.clusters)


def clusters_distance(clusters1, clusters2):
    if len(clusters1) > 0 and len(clusters2) > 0:
        distances = cdist(clusters1.get_clusters_matrix(),
                          clusters2.get_clusters_matrix(), 'cosine')
        return np.amin(distances)
    return 0.5


def clusters_vec_distance(clusters, feature):
    if len(clusters) > 0 and feature is not None:
        distances = cdist(clusters.get_clusters_matrix(),
                          feature.reshape(1, -1), 'cosine')
        return np.amin(distances)
    return 0.5


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
                 track_detection_iou_thresh=0.5):
        self.reid_model = reid_model
        self.global_id_getter = global_id_getter
        self.global_id_releaser = global_id_releaser
        self.id = id
        self.tracks = []
        self.history_tracks = []
        self.time = 0
        self.time_window = time_window
        self.continue_time_thresh = continue_time_thresh
        self.track_clear_thresh = track_clear_thresh
        self.match_threshold = match_threshold
        self.merge_thresh = merge_thresh
        self.n_clusters = n_clusters
        self.max_bbox_velocity = max_bbox_velocity
        self.detection_occlusion_thresh = detection_occlusion_thresh
        self.track_detection_iou_thresh = track_detection_iou_thresh

    def process(self, frame, detections, mask=None):
        reid_features = [None]*len(detections)
        if self.reid_model:
            reid_features = self._get_embeddings(frame, detections, mask)

        assignment = self._continue_tracks(detections, reid_features)
        if self.time % self.time_window == 0:
            self._create_new_tracks(detections, reid_features, assignment)
            self._merge_tracks()
        self.time += 1

    def get_tracked_objects(self):
        objs = []
        label = 'ID'
        objs = []
        for track in self.tracks:
            if track['timestamps'][-1] == self.time - 1 and len(track['timestamps']) > self.time_window:
                objs.append(TrackedObj(track['boxes'][-1],
                                       label + ' ' + str(track['id'])))
        return objs

    def get_tracks(self):
        return self.tracks

    def get_archived_tracks(self):
        return self.history_tracks

    def check_and_merge(self, track_source, track_candidate):
        id_candidate = track_source['id']
        idx = -1
        for i, track in enumerate(self.tracks):
            if track['boxes'] == track_candidate['boxes']:
                idx = i
        if idx < 0:  # in this case track already has been modified, merge is invalid
            return

        collisions_found = False
        for i, track in enumerate(self.tracks):
            if track is not None and track['id'] == id_candidate:
                if self.tracks[i]['timestamps'][-1] <= self.tracks[idx]['timestamps'][0] or \
                            self.tracks[idx]['timestamps'][-1] <= self.tracks[i]['timestamps'][0]:
                    self.tracks[i]['id'] = id_candidate
                    self.tracks[idx]['id'] = id_candidate
                    self._concatenate_tracks(i, idx)
                collisions_found = True

        if not collisions_found:
            self.tracks[idx]['id'] = id_candidate
            new_clusters = self._merge_clustered_features(self.tracks[idx]['f_cluster'],
                                                          track_source['f_cluster'],
                                                          self.tracks[idx]['features'],
                                                          track_source['features'])
            self.tracks[idx]['f_cluster'] = copy(new_clusters)
            track_candidate['f_cluster'] = copy(new_clusters)
        self.tracks = list(filter(None, self.tracks))

    def _continue_tracks(self, detections, features):
        active_tracks_idx = []
        for i, track in enumerate(self.tracks):
            if track['timestamps'][-1] >= self.time - self.continue_time_thresh:
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
                        self._check_velocity_constraint(self.tracks[idx], detections[i]) and \
                        self._giou(self.tracks[idx]['boxes'][-1], detections[i]) > self.track_detection_iou_thresh:
                    assignment[i] = j

            for i, j in enumerate(assignment):
                if j is not None:
                    idx = active_tracks_idx[j]
                    self.tracks[idx]['boxes'].append(detections[i])
                    self.tracks[idx]['timestamps'].append(self.time)
                    self.tracks[idx]['features'].append(features[i])
                    if features[i] is not None:
                        self.tracks[idx]['f_cluster'].update(features[i])
                        if self.tracks[idx]['avg_feature'] is None:
                            self.tracks[idx]['avg_feature'] = np.zeros(features[i].shape)
                        self.tracks[idx]['avg_feature'] += (features[i] - self.tracks[idx]['avg_feature']) / \
                                                            len(self.tracks[idx]['features'])
                    else:
                        self.tracks[idx]['avg_feature'] = None

        return assignment

    def _merge_tracks(self):
        clear_tracks = []
        for track in self.tracks:
            # remove too old tracks
            if track['timestamps'][-1] < self.time - self.track_clear_thresh:
                track['features'] = []
                self.history_tracks.append(track)
                continue
            # remove too short and outdated tracks
            if track['timestamps'][-1] < self.time - self.continue_time_thresh \
                    and len(track['timestamps']) < self.time_window:
                self.global_id_releaser(track['id'])
                continue
            clear_tracks.append(track)
        self.tracks = clear_tracks

        distance_matrix = THE_BIGGEST_DISTANCE * np.eye(len(self.tracks), dtype=np.float32)
        for i, track1 in enumerate(self.tracks):
            for j, track2 in enumerate(self.tracks):
                if j >= i:
                    break
                if (track1['timestamps'][0] > track2['timestamps'][-1] or \
                        track2['timestamps'][0] > track1['timestamps'][-1]) and \
                        len(track1['timestamps']) >= self.time_window and len(track2['timestamps']) >= self.time_window and \
                        track1['avg_feature'] is not None and track2['avg_feature'] is not None:

                        f_avg_dist = cosine(track1['avg_feature'], track2['avg_feature'])
                        f_clust_dist = clusters_distance(track1['f_cluster'], track2['f_cluster'])
                        distance_matrix[i, j] = min(f_avg_dist, f_clust_dist)
                else:
                    distance_matrix[i, j] = THE_BIGGEST_DISTANCE
        distance_matrix += np.transpose(distance_matrix)

        assignment = [None]*distance_matrix.shape[0]
        indices_rows = np.arange(distance_matrix.shape[0])
        indices_cols = np.arange(distance_matrix.shape[1])

        while len(indices_rows) > 0 and len(indices_cols) > 0:
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            dist = distance_matrix[i, j]
            if dist < self.merge_thresh:
                assignment[indices_rows[i]] = indices_cols[j]
                distance_matrix = np.delete(distance_matrix, i, 0)
                distance_matrix = np.delete(distance_matrix, j, 1)
                indices_rows = np.delete(indices_rows, i)
                indices_cols = np.delete(indices_cols, j)
            else:
                break

        for i, idx in enumerate(assignment):
            if idx is not None and self.tracks[idx] is not None and self.tracks[i] is not None:
                self._concatenate_tracks(i, idx)

        self.tracks = list(filter(None, self.tracks))

    def _concatenate_tracks(self, i, idx):
        if self.tracks[i]['timestamps'][-1] <= self.tracks[idx]['timestamps'][0]:
            self.tracks[i]['avg_feature'] = (none_to_zero(self.tracks[i]['avg_feature'])*len(self.tracks[i]['features']) + \
                                             none_to_zero(self.tracks[idx]['avg_feature'])*len(self.tracks[idx]['features']))
            self.tracks[i]['f_cluster'] = self._merge_clustered_features(
                                            self.tracks[i]['f_cluster'],
                                            self.tracks[idx]['f_cluster'],
                                            self.tracks[i]['avg_feature'],
                                            self.tracks[idx]['avg_feature'])
            self.tracks[i]['timestamps'] += self.tracks[idx]['timestamps']
            self.tracks[i]['boxes'] += self.tracks[idx]['boxes']
            self.tracks[i]['features'] += self.tracks[idx]['features']
            self.tracks[i]['avg_feature'] /= len(self.tracks[i]['features'])
            self.tracks[idx] = None
        else:
            assert self.tracks[idx]['timestamps'][-1] <= self.tracks[i]['timestamps'][0]
            self.tracks[idx]['avg_feature'] = (none_to_zero(self.tracks[i]['avg_feature'])*len(self.tracks[i]['features']) + \
                                               none_to_zero(self.tracks[idx]['avg_feature'])*len(self.tracks[idx]['features']))
            self.tracks[idx]['f_cluster'] = self._merge_clustered_features(
                                                self.tracks[i]['f_cluster'],
                                                self.tracks[idx]['f_cluster'],
                                                self.tracks[i]['avg_feature'],
                                                self.tracks[idx]['avg_feature'])
            self.tracks[idx]['timestamps'] += self.tracks[i]['timestamps']
            self.tracks[idx]['boxes'] += self.tracks[i]['boxes']
            self.tracks[idx]['features'] += self.tracks[i]['features']
            self.tracks[idx]['avg_feature'] /= len(self.tracks[idx]['features'])
            self.tracks[i] = None

    def _create_new_tracks(self, detections, features, assignment):
        assert len(detections) == len(features)
        for i, j in enumerate(assignment):
            if j is None:
                self.tracks.append(self._create_tracklet_descr(self.time,
                                                               detections[i],
                                                               self.global_id_getter(),
                                                               features[i]))

    def _create_tracklet_descr(self, timestamp, rect, id, feature):
        return {'id': id,
                'cam_id': self.id,
                'boxes': [rect],
                'timestamps': [timestamp],
                'features': [feature],
                'avg_feature': feature.copy() if feature is not None else None,
                'f_cluster': ClusterFeature(self.n_clusters, feature)}

    def _compute_detections_assignment_cost(self, active_tracks_idx, detections, features):
        affinity_matrix = np.zeros((len(detections), len(active_tracks_idx)), dtype=np.float32)
        for i, idx in enumerate(active_tracks_idx):
            track_box = self.tracks[idx]['boxes'][-1]
            track_avg_feat = self.tracks[idx]['avg_feature']
            for j, d in enumerate(detections):
                iou = 0.5 * self._giou(d, track_box) + 0.5
                if track_avg_feat is not None and features[j] is not None:
                    reid_sim_avg = 1 - cosine(track_avg_feat, features[j])
                    reid_sim_curr = 1 - cosine(self.tracks[idx]['features'][-1], features[j])
                    reid_sim_clust = 1 - clusters_vec_distance(self.tracks[idx]['f_cluster'], features[j])
                    reid_sim = max(reid_sim_avg, reid_sim_curr, reid_sim_clust)
                else:
                    reid_sim = 0.5
                affinity_matrix[j, i] = iou * reid_sim
        return 1 - affinity_matrix

    @staticmethod
    def _area(box):
        return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)

    def _giou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersecion = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                  min(b1[2], b2[2]), min(b1[3], b2[3])])
        enclosing = self._area([min(b1[0], b2[0]), min(b1[1], b2[1]),
                                max(b1[2], b2[2]), max(b1[3], b2[3])])
        u = a1 + a2 - intersecion
        iou = intersecion / u if u > 0 else 0
        giou = iou - (enclosing - u) / enclosing if enclosing > 0 else -1
        return giou

    def _iou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersecion = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                  min(b1[2], b2[2]), min(b1[3], b2[3])])
        u = a1 + a2 - intersecion
        return intersecion / u if u > 0 else 0

    def _ios(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            # intersection over self
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersecion = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                  min(b1[2], b2[2]), min(b1[3], b2[3])])
        m = a1
        return intersecion / m if m > 0 else 0

    def _get_embeddings(self, frame, detections, mask=None):
        rois = []
        embeddings = []
        for i in range(len(detections)):
            rect = detections[i]
            left, top, right, bottom = rect
            crop = frame[top:bottom, left:right]
            if mask and len(mask[i]) > 0:
                crop = cv2.bitwise_and(crop, crop, mask=mask[i])
            if left != right and top != bottom:
                rois.append(crop)

        if rois:
            embeddings = self.reid_model.forward(rois)
            assert len(rois) == len(embeddings)

        return embeddings

    def _merge_clustered_features(self, clusters1, clusters2, features1, features2):
        if len(features1) >= len(features2):
            for feature in features2:
                if feature is not None:
                    clusters1.update(feature)
            return clusters1
        else:
            for feature in features1:
                if feature is not None:
                    clusters2.update(feature)
            return clusters2

    def _check_velocity_constraint(self, track, detection):
        dt = self.time - track['timestamps'][-1]
        h = abs(detection[0] - detection[1])
        w = abs(detection[2] - detection[3])
        size = 0.5*(w + h)
        shifts = [abs(x - y) for x, y in zip(track['boxes'][-1], detection)]
        velocity = sum(shifts) / len(shifts) / dt / size
        if velocity > self.max_bbox_velocity:
            return False
        return True
