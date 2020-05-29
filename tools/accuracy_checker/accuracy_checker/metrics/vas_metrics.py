import numpy as np
from collections import namedtuple

from accuracy_checker.metrics.metric import FullDatasetEvaluationMetric
from accuracy_checker.representation import (
    BaseRepresentation, 
    ReIdentificationClassificationAnnotation, 
    ReIdentificationPrediction
)
from accuracy_checker.config import NumberField, BoolField
from accuracy_checker.presenters import EvaluationResult

Pair = namedtuple('Pair', 'image1 image2 match')

class VASFRLFWMetric(FullDatasetEvaluationMetric):
    __provider__ = 'vas_fr_lfw_metric'

    annotation_types = (ReIdentificationClassificationAnnotation, )
    prediction_types = (ReIdentificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'threshold': NumberField(
                value_type=float, 
                min_value=0, 
                optional=False,
                description='Threshold value to identify pair of faces as matched'
            )
        })
        return parameters

    def regroup_pairs(self, annotations, predictions):
        image_indexes = {}

        for i, pred in enumerate(predictions):
            image_indexes[pred.identifier] = i
            pairs = []

        for image1 in annotations:
            for image2 in image1.positive_pairs:
                if image2 in image_indexes:
                    pairs.append(Pair(image_indexes[image1.identifier], image_indexes[image2], True))
            for image2 in image1.negative_pairs:
                if image2 in image_indexes:
                    pairs.append(Pair(image_indexes[image1.identifier], image_indexes[image2], False))
        
        return pairs

    def configure(self):
        self.threshold = self.get_value_from_config('threshold')
        self.print_verbose = self.get_value_from_config('print_verbose')

    def submit_all(self, annotations, predictions):
        if len(predictions) > 0 and type(predictions[0]) == str:
            return 0   
        return self.evaluate(annotations, predictions)

    def evaluate(self, annotations, predictions):
        tp = fp = tn = fn = 0
        pairs = self.regroup_pairs(annotations, predictions)

        for pair in pairs:
            # Dot product of embeddings
            prediction = np.dot(predictions[pair.image1].embedding, predictions[pair.image2].embedding)

            # Similarity scale-shift
            prediction = (prediction + 1) / 2
            
            # Calculate metrics
            if pair.match: # Pairs that match
                if prediction > self.threshold:
                    tp += 1
                else:
                    fp += 1
            else:
                if prediction < self.threshold:
                    tn += 1
                else:
                    fn += 1

        print('====== Result Summary ======')
        print('Threshold: {}'.format(self.threshold))
        print('TP:', tp)
        print('FP:', fp)
        print('TN:', tn)
        print('FN:', fn)

        return [(tp+tn) / (tp+fp+tn+fn)]