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

from ..representation import (
    RawTensorAnnotation,
    RawTensorPrediction,
)

from .metric import FullDatasetEvaluationMetric
from ..config import NumberField
from ..utils import UnsupportedPackage

try:
    from scipy.linalg import sqrtm
except ImportError as error:
    sqrtm = UnsupportedPackage('scipy.linalg.sqrtm', error)


class BaseGanMetric(FullDatasetEvaluationMetric):
    annotation_types = (RawTensorAnnotation, )
    prediction_types = (RawTensorPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'eps': NumberField(
                optional=True, default=1E-16,
                description="Epsilon to avoid nan during calculations",
                value_type=float
            )
        })
        parameters.update({
            'length': NumberField(
                default=1001, description="Length of input feature vector for metric",
                value_type=int
            )
        })

        return parameters

    def configure(self):
        self.meta.update({
            'scale': 1, 'postfix': ' ', 'target': 'higher-worse'
        })
        self.eps = self.get_value_from_config('eps')
        self.length = self.get_value_from_config('length')

    def score_calc(self, annotations, predictions):
        pass

    def evaluate(self, annotations, predictions):
        annotations = [item.value for item in annotations]
        predictions = [item.value for item in predictions]

        real = [item for item in annotations if item.size == self.length]
        real = np.stack(real)
        generated = [item for item in predictions if item.size == self.length]
        generated = np.stack(generated)
        return self.score_calc(real, generated)


class InceptionScore(BaseGanMetric):
    """
    Class for evaluating inception score of GAN models.
    """

    __provider__ = 'inception_score'

    def score_calc(self, annotations, predictions):
        """
        Calculate IS metric for prediction labels.
        """

        mean_probabilities_of_classes = np.expand_dims(np.mean(predictions, axis=0), axis=0)
        KL_d = predictions * (np.log(predictions + self.eps) - np.log(mean_probabilities_of_classes + self.eps))
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
        if isinstance(sqrtm, UnsupportedPackage):
            sqrtm.raise_error(self.__provider__)

        assert annotations.shape[1] == predictions.shape[1], "Expected equal length of feature vectors"

        mu_real, mu_gen = annotations.mean(axis=0), predictions.mean(axis=0)
        cov_real, cov_gen = np.cov(annotations, rowvar=False), np.cov(predictions, rowvar=False)
        mdiff = mu_real - mu_gen

        covmean = sqrtm(cov_real.dot(cov_gen))
        if not np.isfinite(covmean).all():
            offset = np.eye(cov_real.shape[0]) * self.eps
            covmean = sqrtm((cov_real + offset).dot(cov_gen + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = mdiff.dot(mdiff) + np.trace(cov_real + cov_gen - 2 * covmean)
        return fid
