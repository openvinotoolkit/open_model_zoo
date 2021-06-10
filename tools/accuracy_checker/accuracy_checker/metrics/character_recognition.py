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

from ..representation import CharacterRecognitionAnnotation, CharacterRecognitionPrediction
from .metric import PerImageEvaluationMetric
from .average_meter import AverageMeter
from .average_editdistance_meter import AverageEditdistanceMeter
from ..utils import UnsupportedPackage
from ..config import BoolField
try:
    import editdistance
except ImportError as import_error:
    editdistance = UnsupportedPackage("editdistance", import_error.msg)


class CharacterRecognitionAccuracy(PerImageEvaluationMetric):
    __provider__ = 'character_recognition_accuracy'

    annotation_types = (CharacterRecognitionAnnotation, )
    prediction_types = (CharacterRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'remove_spaces': BoolField(optional=True, default=False)
        })
        return params

    def configure(self):
        self.accuracy = AverageMeter(lambda annotation, prediction: int(annotation == prediction))
        self.remove_spaces = self.get_value_from_config('remove_spaces')

    def update(self, annotation, prediction):
        gt_label = annotation.label
        pred_label = prediction.label
        if self.remove_spaces:
            gt_label = gt_label.replace(' ', '')
            pred_label = pred_label.replace(' ', '')
        return self.accuracy.update(gt_label, pred_label)

    def evaluate(self, annotations, predictions):
        return self.accuracy.evaluate()

    def reset(self):
        self.accuracy.reset()


class LabelLevelRecognitionAccuracy(PerImageEvaluationMetric):
    __provider__ = 'label_level_recognition_accuracy'

    annotation_types = (CharacterRecognitionAnnotation, )
    prediction_types = (CharacterRecognitionPrediction, )

    def configure(self):
        if isinstance(editdistance, UnsupportedPackage):
            editdistance.raise_error(self.__provider__)
        self.accuracy = AverageEditdistanceMeter(editdistance.eval)

    def update(self, annotation, prediction):
        return self.accuracy.update(annotation.label, prediction.label)

    def evaluate(self, annotations, predictions):
        return self.accuracy.evaluate()

    def reset(self):
        self.accuracy.reset()
