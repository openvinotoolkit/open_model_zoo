"""
Copyright (c) 2018-2024 Intel Corporation

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

from ..representation import (
    CharacterRecognitionAnnotation,
    CharacterRecognitionPrediction,
)
from .metric import PerImageEvaluationMetric
from .distance import editdistance_eval


class SpeechRecognitionWER(PerImageEvaluationMetric):
    __provider__ = 'wer'
    annotation_types = (CharacterRecognitionAnnotation,)
    prediction_types = (CharacterRecognitionPrediction,)

    def configure(self):
        self.words = 0
        self.score = 0

    def update(self, annotation, prediction):
        cur_score = editdistance_eval(annotation.label.split(), prediction.label.split())
        cur_words = len(annotation.label.split())
        self.score += cur_score
        self.words += cur_words
        return cur_score / cur_words if cur_words != 0 else 0

    def evaluate(self, annotations, predictions):
        return self.score / self.words if self.words != 0 else 0

    def reset(self):
        self.words, self.score = 0, 0

    @classmethod
    def get_common_meta(cls):
        meta = super().get_common_meta()
        meta['target'] = 'higher-worse'
        return meta


class SpeechRecognitionCER(PerImageEvaluationMetric):
    __provider__ = 'cer'
    annotation_types = (CharacterRecognitionAnnotation,)
    prediction_types = (CharacterRecognitionPrediction,)

    def configure(self):
        self.length = 0
        self.score = 0

    def update(self, annotation, prediction):
        cur_score = editdistance_eval(annotation.label, prediction.label)
        cur_length = len(annotation.label)
        self.score += cur_score
        self.length += cur_length
        return cur_score / cur_length if cur_length != 0 else 0

    def evaluate(self, annotations, predictions):
        return self.score / self.length if self.length != 0 else 0

    def reset(self):
        self.length, self.score = 0, 0

    @classmethod
    def get_common_meta(cls):
        meta = super().get_common_meta()
        meta['target'] = 'higher-worse'
        return meta


class SpeechRecognitionSER(PerImageEvaluationMetric):
    __provider__ = 'ser'

    annotation_types = (CharacterRecognitionAnnotation,)
    prediction_types = (CharacterRecognitionPrediction,)

    def configure(self):
        self.length = 0
        self.score = 0

    def update(self, annotation, prediction):
        # remove extra whitespaces
        gt_label = ' '.join(annotation.label.split())
        pred_label = ' '.join(prediction.label.split())
        ser = int(gt_label != pred_label)
        self.score += ser
        self.length += 1
        return ser

    def evaluate(self, annotations, predictions):
        return self.score / self.length if self.length != 0 else 0

    def reset(self):
        self.length, self.score = 0, 0

    @classmethod
    def get_common_meta(cls):
        meta = super().get_common_meta()
        meta['target'] = 'higher-worse'
        return meta
