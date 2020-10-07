"""
Copyright (c) 2018-2020 Intel Corporation

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

from ..representation import LanguageModelingAnnotation, LanguageModelingPrediction
from .metric import PerImageEvaluationMetric


class ScorePerplexity(PerImageEvaluationMetric):
    __provider__ = 'perplexity'

    annotation_types = (LanguageModelingAnnotation,)
    prediction_types = (LanguageModelingPrediction,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = 0
        self.total = 0
        self.meta['target'] = 'higher-worse'

    def update(self, annotation, prediction):
        def cross_entropy(logits, target):
            return nll_loss(log_softmax(logits, 1), target)

        def log_softmax(x, dim):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return np.log(e_x / e_x.sum(axis=-1, keepdims=True))

        def nll_loss(logs, targets):
            out = logs[range(len(targets)), targets]
            return -out.sum() / out.size

        shift_logits = prediction.logits[:-1, :]
        shift_labels = annotation.labels[1:]
        step_loss = cross_entropy(shift_logits, shift_labels)
        self.loss += step_loss
        self.total += 1
        return step_loss

    def evaluate(self, annotations, predictions):
        if self.total == 0:
            return 0
        return np.exp(self.loss / self.total) / 100

    def reset(self):
        self.loss = 0
        self.total = 0
