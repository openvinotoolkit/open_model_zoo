import numpy as np
from .metric import PerImageEvaluationMetric
from ..representation import LMAnnotation, LMPrediction


class Perplexity(PerImageEvaluationMetric):
    __provider__ = 'perplexity'
    annotation_types = (LMAnnotation, )
    prediction_types = (LMPrediction, )

    def configure(self):
        self.perplexity = []

    def update(self, annotation, prediction):
        sentence_perplexity = []
        for idx, target_value in enumerate(annotation.target_ids):
            prediction_score = prediction.scores[idx, target_value]
            enthropy = -np.log2(prediction_score)
            sample_perplexity = 2 ** enthropy
            sentence_perplexity.append(sample_perplexity)
        self.perplexity.extend(sentence_perplexity)

        return np.mean(sentence_perplexity)

    def evaluate(self, annotations, predictions):
        return np.mean(self.perplexity)
