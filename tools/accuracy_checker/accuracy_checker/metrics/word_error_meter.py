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


class WordErrorMeter:
    def __init__(self, threshold=20.0, counter=None):
        self.counter = counter or (lambda x: 1)
        self.accumulator = None
        self.total_count = None
        self.threshold = threshold

    def update(self, annotation_val, prediction_val):
        anno_words = annotation_val.split()
        pred_words = prediction_val.split()

        anno_nump1 = len(anno_words) + 1
        pred_nump1 = len(pred_words) + 1

        dist = np.zeros((anno_nump1, pred_nump1), dtype=np.uint8)

        for i in range(anno_nump1):
            for j in range(pred_nump1):
                if i == 0:
                    dist[0][j] = j
                elif j == 0:
                    dist[i][0] = i

        for i in range(anno_nump1):
            for j in range(pred_nump1):
                if anno_words[i -1] == pred_words[j -1]:
                    dist[i][j] = dist[i-1][j-1]
                else:
                    _sub = dist[i-1][j-1] + 1   # substitute
                    _ins = dist[i][j-1] + 1     # insert
                    _del = dist[i - 1][j] + 1   # delete
                    dist[i][j] = min(_sub, _ins, _del)

        _WER = float(dist[anno_nump1 -1][pred_nump1 - 1]) / (anno_nump1 -1) * 100

        loss = int(_WER <= self.threshold)

        increment = self.counter(annotation_val)

        if self.accumulator is None and self.total_count is None:
            # wrap in array for using numpy.divide with where attribute
            # and support cases when loss function returns list-like object
            self.accumulator = np.array(loss, dtype=float)
            self.total_count = np.array(increment, dtype=float)
        else:
            self.accumulator += loss
            self.total_count += increment

    def evaluate(self):
        if self.total_count is None:
            return 0.0

        return np.divide(
            self.accumulator, self.total_count, out=np.zeros_like(self.accumulator), where=self.total_count != 0
        )
