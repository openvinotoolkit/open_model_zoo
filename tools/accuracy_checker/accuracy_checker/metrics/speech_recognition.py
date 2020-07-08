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

from ..config import BoolField, NumberField, ConfigError
from ..representation import (
    CharacterRecognitionAnnotation,
    CharacterRecognitionPrediction,
    )
from .metric import PerImageEvaluationMetric
from ..utils import finalize_metric_result


class SpeechRecognitionWER(PerImageEvaluationMetric):
    __provider__ = 'wer'
    annotation_types = (CharacterRecognitionAnnotation,)
    prediction_types = (CharacterRecognitionPrediction,)

    overall_metric = []

    def configure(self):
        self.overall_metric = []

    def update(self, annotation, prediction):

        # distance = editDistance(annotation, prediction)
        h = prediction
        r = annotation

        dist = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8).reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            dist[i][0] = i
        for j in range(len(h) + 1):
            dist[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    dist[i][j] = dist[i - 1][j - 1]
                else:
                    substitute = dist[i - 1][j - 1] + 1
                    insert = dist[i][j - 1] + 1
                    delete = dist[i - 1][j] + 1
                    dist[i][j] = min(substitute, insert, delete)

        # print the result in aligned way
        result = float(d[len(r)][len(h)]) / len(r)

        self.overall_metric.append(result)

        return result

    def evaluate(self, annotations, predictions):
        result = np.mean(self.overall_metric, axis=0)
        return result

    def reset(self):
        self.overall_metric = []
