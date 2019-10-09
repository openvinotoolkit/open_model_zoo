######################################
#
#   John Feng, john.feng@intel.com
#
#   using word error rate to evaluate
#   deep speech
#
######################################

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
