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

from .metric import PerImageEvaluationMetric
from ..config import NumberField, StringField, BoolField, ConfigError
from ..representation import (
    ElectricityTimeSeriesForecastingAnnotation,
    ElectricityTimeSeriesForecastingPrediction
)


def normalised_quantile_loss(y, y_pred, quantile):
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
        + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

    quantile_loss = weighted_errors.mean()
    normaliser = np.abs(y).mean()

    return 2 * quantile_loss / normaliser


class NormalisedQuantileLoss(PerImageEvaluationMetric):
    """
    Class for evaluating accuracy metric of normalised quantile loss.
    """

    __provider__ = 'normalised_quantile_loss'

    annotation_types = (ElectricityTimeSeriesForecastingAnnotation, )
    prediction_types = (ElectricityTimeSeriesForecastingPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            "quantile": NumberField(
                description='Quantile for evaluation.', value_type=float
            )
        })
        return parameters

    def configure(self):
        self.quantile = float(self.get_value_from_config('quantile'))
        self.reset()
        super().configure()

    def update(self, annotation, prediction):
        self.annotations.append(annotation)
        self.predictions.append(prediction)

    def evaluate(self, annotations, predictions):
        preds, gt = [], []
        for i in range(len(self.annotations)):
            gt.append(
                self.annotations[i].scaler.inverse_transform(
                    self.annotations[i].outputs[:, :, 0]
                )
            )
            preds.append(
                self.annotations[i].scaler.inverse_transform(
                    self.predictions[i].preds[self.quantile]
                )
            )
        gt = np.concatenate(gt, axis=0)
        preds = np.concatenate(preds, axis=0)
        loss = normalised_quantile_loss(gt, preds, self.quantile)
        print(f"loss: {loss}")
        return loss

    def reset(self):
        self.annotations = []
        self.predictions = []
