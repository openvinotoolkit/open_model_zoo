import numpy as np
from .metric import PerImageEvaluationMetric
from ..representation import LMAnnotation, LMPrediction


class Perplexity(PerImageEvaluationMetric):
    __provider__ = 'perplexity'
    annotation_types = (LMAnnotation, )
    prediction_types = (LMPrediction, )

    def configure(self):
        self.perplexity = []

    @staticmethod
    def _cross_enthropy(scores, targets):
        return np.sum(np.log(scores[range(len(targets)), targets]))

    def update(self, annotation, prediction):
        sentence_perplexity = np.exp(self._cross_enthropy(prediction.scores, annotation.target_ids))
        self.perplexity.append(sentence_perplexity)
        return sentence_perplexity

    def evaluate(self, annotations, predictions):
        return np.mean(self.perplexity)
