"""
Copyright (c) 2018-2021 Intel Corporation

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

from .base_representation import BaseRepresentation
from ..utils import softmax


class Classification(BaseRepresentation):
    pass


class ClassificationAnnotation(Classification):
    def __init__(self, identifier='', label=None):
        super().__init__(identifier)

        self.label = label


class ClassificationPrediction(Classification):
    def __init__(self, identifier='', scores=None):
        super().__init__(identifier)

        self.scores = np.array(scores) if scores is not None else np.array([])

    @property
    def label(self):
        return np.argmax(self.scores)

    def top_k(self, k):
        return np.argpartition(self.scores, -k)[-k:]

    def to_annotation(self, **kwargs):
        scores = softmax(self.scores) if self.scores.max() > 1.0 or self.scores.min() < 0.0 else self.scores
        threshold = kwargs.get('threshold', 0.0)
        if scores.max() > threshold:
            return ClassificationAnnotation(self.identifier, self.label)
        return None


class ArgMaxClassificationPrediction(ClassificationPrediction):
    def __init__(self, identifier='', label=None):
        super().__init__(identifier)
        self._label = label

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    def top_k(self, k):
        return np.full(k, self._label)


class SequenceClassificationAnnotation(ClassificationAnnotation):
    pass


class SequenceClassificationPrediction(ClassificationPrediction):
    @property
    def label(self):
        return np.argmax(self.scores, axis=1)

    def top_k(self, k):
        return np.argpartition(self.scores, -k, axis=1)[:, -k:]
