import numpy as np

from ..representation import (
    ClassificationAnnotation,
    ClassificationPrediction,
    TextClassificationAnnotation,
    UrlClassificationAnnotation,
    ArgMaxClassificationPrediction,
    AnomalySegmentationAnnotation,
    AnomalySegmentationPrediction
)

from .classification import ClassificationProfilingSummaryHelper

from ..config import NumberField, StringField, ConfigError, BoolField
from .metric import Metric, PerImageEvaluationMetric
from .average_meter import AverageMeter
from ..utils import UnsupportedPackage

try:
    from sklearn.metrics import accuracy_score, confusion_matrix
except ImportError as import_error:
    accuracy_score = UnsupportedPackage("sklearn.metric.accuracy_score", import_error.msg)
    confusion_matrix = UnsupportedPackage("sklearn.metric.confusion_matrix", import_error.msg)



class ClassificationGraphAccuracy(PerImageEvaluationMetric):
    """
    Class for evaluating accuracy metric of classification models.
    """

    __provider__ = 'node_accuracy'

    annotation_types = (ClassificationAnnotation, TextClassificationAnnotation)
    prediction_types = (ClassificationPrediction, ArgMaxClassificationPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'top_k': NumberField(
                value_type=int, min_value=1, optional=True, default=1,
                description="The number of classes with the highest probability, which will be used to decide "
                            "if prediction is correct."
            ),
            'match': BoolField(optional=True, default=False),
            'cast_to_int': BoolField(optional=True, default=False)
        })

        return parameters

    def configure(self):
        self.top_k = self.get_value_from_config('top_k')
        self.match = self.get_value_from_config('match')
        self.cast_to_int = self.get_value_from_config('cast_to_int')
        self.summary_helper = None

        def loss(annotation_label, prediction_top_k_labels):
            return int(annotation_label in prediction_top_k_labels)

        if isinstance(accuracy_score, UnsupportedPackage):
            accuracy_score.raise_error(self.__provider__)
        self.accuracy = []
        if self.profiler:
            self.summary_helper = ClassificationProfilingSummaryHelper()

    def set_profiler(self, profiler):
        self.profiler = profiler
        self.summary_helper = ClassificationProfilingSummaryHelper()

    def update(self, annotation, prediction):
        pred_labels = prediction.scores

        accuracy = accuracy_score(annotation.label, pred_labels)
        self.accuracy.append(accuracy)

        if self.profiler:
            self.summary_helper.submit_data(annotation.label, prediction.top_k(self.top_k), prediction.scores)
            self.profiler.update(
                annotation.identifier, annotation.label, prediction.top_k(self.top_k), self.name, accuracy,
                prediction.scores
            )
        return accuracy

    def evaluate(self, annotations, predictions):
        if self.profiler:
            self.profiler.finish()
            summary = self.summary_helper.get_summary_report()
            self.profiler.write_summary(summary)
        else:
            accuracy = np.mean(self.accuracy)
        return accuracy

    def reset(self):
        if not self.match:
            self.accuracy.reset()
        else:
            self.accuracy = []

        if self.profiler:
            self.profiler.reset()