"""
Copyright (c) 2018-2022 Intel Corporation

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

from ..utils import remove_difficult
from .base_representation import BaseRepresentation


class Detection(BaseRepresentation):
    def __init__(self, identifier='', labels=None, x_mins=None, y_mins=None, x_maxs=None, y_maxs=None, metadata=None):
        super().__init__(identifier, metadata)

        self.labels = np.array(labels) if labels is not None else np.array([])
        self.x_mins = np.array(x_mins) if x_mins is not None else np.array([])
        self.y_mins = np.array(y_mins) if y_mins is not None else np.array([])
        self.x_maxs = np.array(x_maxs) if x_maxs is not None else np.array([])
        self.y_maxs = np.array(y_maxs) if y_maxs is not None else np.array([])

    def remove(self, indexes):
        self.labels = np.delete(self.labels, indexes)
        self.x_mins = np.delete(self.x_mins, indexes)
        self.y_mins = np.delete(self.y_mins, indexes)
        self.x_maxs = np.delete(self.x_maxs, indexes)
        self.y_maxs = np.delete(self.y_maxs, indexes)

        difficult_boxes = self.metadata.get('difficult_boxes')
        if not difficult_boxes:
            return

        new_difficult_boxes = remove_difficult(difficult_boxes, indexes)

        self.metadata['difficult_boxes'] = new_difficult_boxes

    @property
    def size(self):
        return len(self.x_mins)

    @property
    def boxes(self):
        if self.size == 0:
            return []

        return [[x_min, y_min, x_max, y_max]
                for x_min, y_min, x_max, y_max in zip(self.x_mins, self.y_mins, self.x_maxs, self.y_maxs)]

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        def are_bounding_boxes_equal():
            if not np.array_equal(self.labels, other.labels):
                return False
            if not np.array_equal(self.x_mins, other.x_mins):
                return False
            if not np.array_equal(self.y_mins, other.y_mins):
                return False
            if not np.array_equal(self.x_maxs, other.x_maxs):
                return False
            if not np.array_equal(self.y_maxs, other.y_maxs):
                return False
            return True

        return self.identifier == other.identifier and are_bounding_boxes_equal() and self.metadata == other.metadata


class DetectionAnnotation(Detection):
    pass


class DetectionPrediction(Detection):
    def __init__(self, identifier='', labels=None, scores=None, x_mins=None, y_mins=None, x_maxs=None, y_maxs=None,
                 metadata=None):
        super().__init__(identifier, labels, x_mins, y_mins, x_maxs, y_maxs, metadata)
        self.scores = np.array(scores) if scores is not None else np.array([])

    def remove(self, indexes):
        super().remove(indexes)
        self.scores = np.delete(self.scores, indexes)

    def __eq__(self, other):
        return np.array_equal(self.scores, other.scores) if super().__eq__(other) else False

    def to_annotation(self, **kwargs):
        return DetectionAnnotation(
            self.identifier, self.labels, self.x_mins, self.y_mins, self.x_maxs, self.y_maxs, self.metadata
        )

    def replace(self, labels, scores, x_mins, y_mins, x_maxs, y_maxs):
        self.remove(np.array(range(self.scores.size)))
        self.labels = np.array(labels) if labels is not None else np.array([])
        self.x_mins = np.array(x_mins) if x_mins is not None else np.array([])
        self.y_mins = np.array(y_mins) if y_mins is not None else np.array([])
        self.x_maxs = np.array(x_maxs) if x_maxs is not None else np.array([])
        self.y_maxs = np.array(y_maxs) if y_maxs is not None else np.array([])
        self.scores = np.array(scores) if scores is not None else np.array([])

class AttributeDetectionAnnotation(DetectionAnnotation):
    pass


class AttributeDetectionPrediction(DetectionPrediction):
    def __init__(
            self,
            identifier='',
            labels=None,
            scores=None,
            bbox_scores=None,
            x_mins=None,
            y_mins=None,
            x_maxs=None,
            y_maxs=None
    ):
        super().__init__(identifier, labels, scores, x_mins, y_mins, x_maxs, y_maxs)
        self.bbox_scores = np.array(bbox_scores) if bbox_scores is not None else np.array([])

    def remove(self, indexes):
        super().remove(indexes)
        self.bbox_scores = np.delete(self.bbox_scores, indexes)

    def __eq__(self, other):
        return np.array_equal(self.bbox_scores, other.bbox_scores) if super().__eq__(other) else False

    def to_annotation(self, **kwargs):
        return AttributeDetectionAnnotation(
            self.identifier, self.labels, self.x_mins, self.y_mins, self.x_maxs, self.y_maxs, self.metadata
        )

class ActionDetectionAnnotation(AttributeDetectionAnnotation):
    pass


class ActionDetectionPrediction(AttributeDetectionPrediction):
    pass
