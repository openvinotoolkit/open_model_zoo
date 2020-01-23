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
from .base_representation import BaseRepresentation
from .classification_representation import ClassificationPrediction


class LMRepresentation(BaseRepresentation):
    pass


class LMAnnotation(LMRepresentation):
    def __init__(self, identifier, input_ids, target_ids, input_words=None, target_words=None, metadata=None):
        super().__init__(identifier, metadata)
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.input_words = input_words
        self.target_words = target_words


class LMPrediction(LMRepresentation, ClassificationPrediction):
    @property
    def label(self):
        return np.argmax(self.scores, axis=0)

    def top_k(self, k):
        return np.argpartition(self.scores, -k, axis=0)[:, -k:]
