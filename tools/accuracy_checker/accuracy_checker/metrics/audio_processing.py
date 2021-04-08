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
from ..config import NumberField
from ..representation import NoiseSuppressionAnnotation, NoiseSuppressionPrediction


class SISDRMetric(PerImageEvaluationMetric):
    __provider__ = 'sisdr'
    annotation_types = (NoiseSuppressionAnnotation, )
    prediction_types = (NoiseSuppressionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'delay': NumberField(
                optional=True, default=0, value_type=int, min_value=0,
                description='shift output by delay to align with reference'
            )
        })
        return parameters

    def configure(self):
        self.delay = self.get_value_from_config('delay')
        self.buffer = []
        self.meta.update({'scale': 1, 'postfix': 'Db', 'calculate_mean': False, 'names': ['mean', 'std']})
        self.meta['target_per_value'] = {'mean': 'higher-better', 'std': 'higher-worse'}

    def reset(self):
        del self.buffer
        self.buffer = []

    def update(self, annotation, prediction):
        target = annotation.value[:-self.delay] # pylint: disable=E1130
        y = prediction.value[self.delay:]

        target = target - np.mean(target, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)

        y_by_target = np.sum(y * target, axis=-1, keepdims=True)
        t2 = np.sum(target ** 2, axis=-1, keepdims=True)
        y_target = y_by_target * target / (t2 + np.finfo(float).eps)
        y_noise = y - y_target

        target_pow = np.sum(y_target ** 2, axis=-1)
        noise_pow = np.sum(y_noise ** 2, axis=-1)

        sisdr = 10 * np.log10(target_pow + np.finfo(float).eps) - 10 * np.log10(noise_pow + np.finfo(float).eps)
        self.buffer.append(sisdr)

        return sisdr

    def evaluate(self, annotations, predictions):
        return [np.mean(self.buffer), np.std(self.buffer)]
