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
import editdistance

from ..representation import (
    CharacterRecognitionAnnotation,
    CharacterRecognitionPrediction,
    )
from .metric import PerImageEvaluationMetric, FullDatasetEvaluationMetric


class SpeechRecognitionWER(PerImageEvaluationMetric, FullDatasetEvaluationMetric):
    __provider__ = 'wer'
    annotation_types = (CharacterRecognitionAnnotation,)
    prediction_types = (CharacterRecognitionPrediction,)

    def configure(self):
        self.overall_metric = []
        self.meta['target'] = 'higher-worse'
        self.words = 0
        self.score = 0

    def update(self, annotation, prediction):

        h = prediction.label
        r = annotation.label
        # print(annotation.identifier)
        # print("p: {}".format(h))
        # print('a: {}'.format(r))
        h_list = h.split()
        r_list = r.split()
        self.words += len(r_list)
        self.score += editdistance.eval(h_list, r_list)

        # dist = np.zeros((len(r) + 1, len(h) + 1), dtype=np.uint8)
        # for i in range(len(r) + 1):
        #     dist[i][0] = i
        # for j in range(len(h) + 1):
        #     dist[0][j] = j
        # for i in range(1, len(r) + 1):
        #     for j in range(1, len(h) + 1):
        #         if r[i - 1] == h[j - 1]:
        #             dist[i][j] = dist[i - 1][j - 1]
        #         else:
        #             substitute = dist[i - 1][j - 1] + 1
        #             insert = dist[i][j - 1] + 1
        #             delete = dist[i - 1][j] + 1
        #             dist[i][j] = min(substitute, insert, delete)
        #
        # result = float(dist[len(r)][len(h)]) / len(r)

        #self.overall_metric.append(result)

        return self.score / self.words

    def evaluate(self, annotations, predictions):
        for ann, pred in zip(annotations, predictions):
            print("a: {}".format(ann.label.lower()))
            print('p: {}'.format(pred.label.lower()))
        return self.score / self.words

    def reset(self):
        self.overall_metric = []
