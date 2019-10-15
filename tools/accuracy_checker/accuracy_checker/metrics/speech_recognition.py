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

###################################################
#
#   John Feng, john.feng@intel.com
#
#   using word error rate to evaluate deep speech
#
####################################################

from ..representation import CharacterRecognitionAnnotation, CharacterRecognitionPrediction
from .metric import PerImageEvaluationMetric
from .word_error_meter import WordErrorMeter


class SpeechRecognitionAccuracy(PerImageEvaluationMetric):
    __provider__ = 'speech_recognition_accuracy'

    annotation_types = (CharacterRecognitionAnnotation, )
    prediction_types = (CharacterRecognitionPrediction, )

    def configure(self):
        self.threshold = self.get_value_from_config('threshold')
        self.accuracy = WordErrorMeter(self.threshold)

    def update(self, annotation, prediction):
        self.accuracy.update(annotation.label, prediction.label)

    def evaluate(self, annotations, predictions):
        return self.accuracy.evaluate()
