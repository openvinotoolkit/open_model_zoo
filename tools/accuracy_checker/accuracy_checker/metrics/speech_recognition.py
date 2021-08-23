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

from ..representation import (
    CharacterRecognitionAnnotation,
    CharacterRecognitionPrediction,
)
from .metric import PerImageEvaluationMetric
from ..utils import UnsupportedPackage

try:
    import editdistance
except ImportError as import_error:
    editdistance = UnsupportedPackage("editdistance", import_error.msg)


class SpeechRecognitionWER(PerImageEvaluationMetric):
    __provider__ = 'wer'
    annotation_types = (CharacterRecognitionAnnotation,)
    prediction_types = (CharacterRecognitionPrediction,)

    def configure(self):
        if isinstance(editdistance, UnsupportedPackage):
            editdistance.raise_error(self.__provider__)
        self.words = 0
        self.score = 0
        self.meta['target'] = 'higher-worse'

    def update(self, annotation, prediction):
        cur_score = editdistance.eval(annotation.label.split(), prediction.label.split())
        cur_words = len(annotation.label.split())
        self.score += cur_score
        self.words += cur_words
        return cur_score / cur_words

    def evaluate(self, annotations, predictions):
        return self.score / self.words if self.words != 0 else 0

    def reset(self):
        self.words, self.score = 0, 0


class SpeechRecognitionCER(PerImageEvaluationMetric):
    __provider__ = 'cer'
    annotation_types = (CharacterRecognitionAnnotation,)
    prediction_types = (CharacterRecognitionPrediction,)

    def configure(self):
        if isinstance(editdistance, UnsupportedPackage):
            editdistance.raise_error(self.__provider__)
        self.length = 0
        self.score = 0
        self.meta['target'] = 'higher-worse'

    def update(self, annotation, prediction):
        cur_score = editdistance.eval(annotation.label, prediction.label)
        cur_length = len(annotation.label)
        self.score += cur_score
        self.length += cur_length
        return cur_score / cur_length

    def evaluate(self, annotations, predictions):
        return self.score / self.length if self.length != 0 else 0

    def reset(self):
        self.length, self.score = 0, 0


class SpeechRecognitionSER(PerImageEvaluationMetric):
    __provider__ = 'ser'

    annotation_types = (CharacterRecognitionAnnotation,)
    prediction_types = (CharacterRecognitionPrediction,)

    def configure(self):
        self.length = 0
        self.score = 0
        self.meta['target'] = 'higher-worse'

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
