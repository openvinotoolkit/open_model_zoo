"""
Copyright (c) 2018-2024 Intel Corporation

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

import cv2
import numpy as np


from ..representation import (
    BackgroundMattingAnnotation,
    BackgroundMattingPrediction
)

from ..config import BoolField, NumberField, StringField
from .metric import PerImageEvaluationMetric
from ..utils import UnsupportedPackage

try:
    from scipy.ndimage.filters import convolve
except ImportError as err:
    convolve = UnsupportedPackage('scipy', err.msg)


class BaseBackgroundMattingMetrics(PerImageEvaluationMetric):
    annotation_types = (BackgroundMattingAnnotation,)
    prediction_types = (BackgroundMattingPrediction,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'process_type': StringField(
                choices=cls.all_provided_process_types(), optional=True, default='',
                description="Specifies method of processing that will be used."
            )
        })
        return parameters

    @staticmethod
    def all_provided_process_types():
        return ['alpha', 'image']

    @staticmethod
    def prepare_pha(annotation):
        image = annotation.value
        if image.shape[-1] == 4:
            image = image[:, :, -1]
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image.astype(np.float32) / 255

    @staticmethod
    def prepare_fgr(annotation):
        image = annotation.fgr
        if image.shape[-1] == 4:
            image = image[:, :, :-1]
        return image.astype(np.float32) / 255

    def get_annotation(self, annotation):
        return self.process_func(annotation)

    def get_prediction(self, prediction):
        return prediction.value[self.prediction_source]

    def evaluate(self, annotations, predictions):
        return sum(self.results) / len(self.results)

    def reset(self):
        self.results = []

    def configure(self):
        process_type = self.get_value_from_config('process_type')
        self.process_func = self.prepare_pha if process_type == 'alpha' else self.prepare_fgr
        self.reset()

    @classmethod
    def get_common_meta(cls):
        return {'target': 'higher-worse', 'scale': 1, 'postfix': ' '}


class MeanOfAbsoluteDifference(BaseBackgroundMattingMetrics):
    __provider__ = 'mad'

    def update(self, annotation, prediction):
        pred = self.get_prediction(prediction)
        gt = self.get_annotation(annotation)
        if pred.shape[-1] == 1 and pred.shape[-1] != gt.shape[-1]:
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        value = np.mean(abs(pred - gt)) * 1e3
        self.results.append(value)
        return value


class SpatialGradient(BaseBackgroundMattingMetrics):
    __provider__ = 'spatial_gradient'

    def update(self, annotation, prediction):
        pred = self.get_prediction(prediction)
        gt = self.get_annotation(annotation)
        if pred.shape[-1] == 1 and pred.shape[-1] != gt.shape[-1]:
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        gt_grad = self.gauss_gradient(gt)
        pred_grad = self.gauss_gradient(pred)
        value = np.sum((gt_grad - pred_grad) ** 2) / 1000
        self.results.append(value)
        return value

    def gauss_gradient(self, img):
        img_filtered_x = convolve(img, self.filter_x[:, ::-1], mode='reflect')
        img_filtered_y = convolve(img, self.filter_y[::-1, :], mode='reflect')
        return np.sqrt(img_filtered_x ** 2 + img_filtered_y ** 2)

    def gauss_filter(self, sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int32(2 * half_size + 1)
        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = self.gaussian(i - half_size, sigma) * self.dgaussian(
                    j - half_size, sigma)
        # normalize filter
        norm = np.sqrt((filter_x ** 2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)
        return filter_x, filter_y

    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    def dgaussian(self, x, sigma):
        return -x * self.gaussian(x, sigma) / sigma ** 2

    def configure(self, sigma=1.4):
        if isinstance(convolve, UnsupportedPackage):
            convolve.raise_error(self.__provider__)
        super().configure()
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
        self.reset()


class MeanSquaredErrorWithMask(BaseBackgroundMattingMetrics):
    __provider__ = 'mse_with_mask'

    def update(self, annotation, prediction):
        pred = self.get_prediction(prediction)
        gt = self.get_annotation(annotation)
        if pred.shape[-1] == 1 and pred.shape[-1] != gt.shape[-1]:
            gt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        if self.use_mask:
            mask = self.prepare_pha(annotation) > self.pha_tolerance
            pred = pred[mask]
            gt = gt[mask]
        value = np.mean((pred - gt) ** 2) * 1e3
        self.results.append(value)
        return value

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'use_mask': BoolField(
                optional=True, default=False, description="Apply alpha mask to foreground."
            ),
            'pha_tolerance': NumberField(
                optional=True, default=0.01, description="Tolerance to get binary mask from pha.",
                min_value=0.0, max_value=1.0
            )
        })
        return parameters

    def configure(self):
        super().configure()
        self.use_mask = self.get_value_from_config('use_mask')
        self.pha_tolerance = self.get_value_from_config('pha_tolerance')
