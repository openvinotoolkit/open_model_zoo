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

from ..representation import CharacterRecognitionAnnotation, CharacterRecognitionPrediction
from ..config import ConfigError
from .metric import PerImageEvaluationMetric
from .average_meter import AverageMeter
from .average_editdistance_meter import AverageEditdistanceMeter
try:
    import editdistance
except ImportError:
    editdistance = None


class CharacterRecognitionAccuracy(PerImageEvaluationMetric):
    __provider__ = 'character_recognition_accuracy'

    annotation_types = (CharacterRecognitionAnnotation, )
    prediction_types = (CharacterRecognitionPrediction, )

    def configure(self):
        self.accuracy = AverageMeter(lambda annotation, prediction: int(annotation == prediction))

    def update(self, annotation, prediction):
        return self.accuracy.update(annotation.label, prediction.label)


    def evaluate(self, annotations, predictions):
        return self.accuracy.evaluate()

    def reset(self):
        self.accuracy.reset()


class LabelLevelRecognitionAccuracy(PerImageEvaluationMetric):
    __provider__ = 'label_level_recognition_accuracy'

    annotation_types = (CharacterRecognitionAnnotation, )
    prediction_types = (CharacterRecognitionPrediction, )

    def configure(self):
        if editdistance is None:
            raise ConfigError('Metric {} requires editdistance package installation. Please install it.'.format(
                self.__provider__
            ))
        self.accuracy = AverageEditdistanceMeter(editdistance.eval)

    def update(self, annotation, prediction):
        return self.accuracy.update(annotation.label, prediction.label)

    def evaluate(self, annotations, predictions):
        return self.accuracy.evaluate()

    def reset(self):
        self.accuracy.reset()
