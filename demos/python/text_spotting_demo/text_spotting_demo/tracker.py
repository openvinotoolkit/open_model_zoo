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


class StaticIOUTracker(object):
    def __init__(self, iou_threshold=0.5, age_threshold=10):
        super().__init__()
        self.history = []
        self.history_areas = []
        self.history_classes = []
        self.ids = []
        self.age = []
        self.iou_threshold = iou_threshold
        self.age_threshold = age_threshold
        self.last_id = 0

    def affinity(self, masks, classes):
        areas = [np.count_nonzero(mask) for mask in masks]
        affinity_matrix = np.zeros((len(masks), len(self.history)), dtype=np.float32)
        for i, (history_mask, history_area, history_class) in \
                enumerate(zip(self.history, self.history_areas, self.history_classes)):
            for j, (mask, area, cls) in enumerate(zip(masks, areas, classes)):
                if cls != history_class:
                    continue
                intersection = np.count_nonzero(np.logical_and(history_mask, mask))
                union = history_area + area - intersection
                iou = intersection / union
                affinity_matrix[j, i] = iou
        return affinity_matrix, areas

    def __call__(self, masks, classes):
        # Get affinity with history.
        affinity_matrix, areas = self.affinity(masks, classes)

        # Make assignment of currents masks to existing tracks.
        assignment = []
        indices = np.arange(len(self.history))
        for i in range(len(masks)):
            j = 0
            affinity_score = -1.0
            if affinity_matrix.shape[1] > 0:
                j = np.argmax(affinity_matrix[i])
                affinity_score = affinity_matrix[i, j]
            if affinity_score > self.iou_threshold:
                assignment.append(indices[j])
                affinity_matrix = np.delete(affinity_matrix, j, 1)
                indices = np.delete(indices, j)
            else:
                assignment.append(None)

        # Increase age for existing tracks.
        for i in range(len(self.age)):
            self.age[i] += 1

        # Update existing tracks.
        for i, j in enumerate(assignment):
            if j is not None:
                self.history[j] = masks[i]
                self.history_areas[j] = areas[i]
                self.age[j] = 0
                assignment[i] = self.ids[j]

        # Prune out too old tracks.
        alive = tuple(i for i, age in enumerate(self.age) if age < self.age_threshold)
        self.history = list(self.history[i] for i in alive)
        self.history_areas = list(self.history_areas[i] for i in alive)
        self.history_classes = list(self.history_classes[i] for i in alive)
        self.age = list(self.age[i] for i in alive)
        self.ids = list(self.ids[i] for i in alive)

        # Save new tracks.
        for i, j in enumerate(assignment):
            if j is None:
                self.history.append(masks[i])
                self.history_areas.append(areas[i])
                self.history_classes.append(classes[i])
                self.age.append(0)
                self.ids.append(self.last_id)
                assignment[i] = self.last_id
                self.last_id += 1

        return assignment
