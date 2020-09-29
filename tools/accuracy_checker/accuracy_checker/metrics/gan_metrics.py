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

import numpy as np

from ..representation import (
    ImageProcessingAnnotation,
    ImageProcessingPrediction,
)

from .metric import FullDatasetEvaluationMetric
from ..config import NumberField


class BaseGanMetric(FullDatasetEvaluationMetric):
    annotation_types = (ImageProcessingAnnotation, )
    prediction_types = (ImageProcessingPrediction, )

    def configure(self):
        self.meta.update({
            'scale': 1, 'postfix': ' ', 'target': 'higher-worse'
        })

    def get_values(self, representation):
        items = [item.value for item in representation]
        return items

    def score_calc(self, annotations, predictions):
        pass

    def evaluate(self, annotations, predictions):
        annotations = self.get_values(annotations)
        predictions = self.get_values(predictions)
        return self.score_calc(annotations, predictions)


class InceptionScore(BaseGanMetric):
    """
    Class for evaluating inception score of GAN models.
    """

    __provider__ = 'inception_score'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'eps': NumberField(
                optional=True, default=1E-16,
                description="Epsilon to avoid nan when trying to calculate the log of a zero probability",
                value_type=float
            )
        })

    def configure(self):
        super().configure()
        self.eps = self.get_value_from_config('eps')

    def score_calc(self, annotations, predictions):
        """
        Calculate IS metric for prediction labels.
        """

        probabilities = np.stack(predictions)
        mean_probabilities_of_classes = np.expand_dims(np.mean(probabilities, axis=0), axis=0)
        KL_d = probabilities * (np.log(probabilities + self.eps) - np.log(mean_probabilities_of_classes + self.eps))
        KL_D = KL_d.sum(axis=1)
        score = np.exp(np.mean(KL_D))
        return score


class FrechetInceptionDistance(BaseGanMetric):
    """
    Class for evaluating Frechet Inception Distance of GAN models.
    """

    __provider__ = 'fid'

    def score_calc(self, annotations, predictions):
        """
        Calculate FID between feature vector of the real and generated images.
        """
        real = np.stack(annotations)
        generated = np.stack(predictions)

        assert real.shape[1] == generated.shape[1], "Expected equal length of feature vectors"

        mu_real, mu_gen = real.mean(axis=0), generated.mean(axis=0)
        cov_real, cov_gen = real.cov(rowvar=False), generated.cov(rowvar=False)
        mdiff = np.sum((mu_real - mu_gen)**2)
        cov = np.sqrt(cov_real.dot(cov_gen))
        FID = mdiff + np.trace(cov_real + cov_gen - 2 * cov)
        return FID
