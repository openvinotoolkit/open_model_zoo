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

import numpy as np

from .metric import FullDatasetEvaluationMetric
from ..representation import (
    TimeSeriesForecastingAnnotation,
    TimeSeriesForecastingQuantilesPrediction
)


def normalised_quantile_loss(gt, pred, quantile):
    prediction_underflow = gt - pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.) + (1. - quantile) * np.maximum(
        -prediction_underflow, 0.)

    quantile_loss = weighted_errors.mean()
    normaliser = np.abs(gt).mean()

    return 2 * quantile_loss / normaliser


class NormalisedQuantileLoss(FullDatasetEvaluationMetric):
    """
    Class for evaluating accuracy metric of normalised quantile loss.
    """

    __provider__ = 'normalised_quantile_loss'

    annotation_types = (TimeSeriesForecastingAnnotation, )
    prediction_types = (TimeSeriesForecastingQuantilesPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        return parameters

    def configure(self):
        self.meta.update({
            'scale': 1, 'postfix': ' ', 'target': 'higher-worse'
        })
        super().configure()

    def evaluate(self, annotations, predictions):
        quantiles = list(predictions[0].preds.keys())
        quantiles.sort()
        self.meta.update({"names": quantiles, "calculate_mean": False})
        gt = [annotation.outputs for annotation in annotations]
        gt = np.concatenate(gt, axis=0)
        values = []
        for q in quantiles:
            preds = [prediction.preds[q] for prediction in predictions]
            preds = np.concatenate(preds, axis=0)
            loss = normalised_quantile_loss(gt, preds, q)
            values.append(loss)
        return values
