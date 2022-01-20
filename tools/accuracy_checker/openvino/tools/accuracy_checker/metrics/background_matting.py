"""
Copyright (c) 2018-2022 Intel Corporation

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
from scipy.ndimage.filters import convolve

from ..representation import (
    BackgroundMattingAnnotation,
    BackgroundMattingPrediction
)

from .metric import PerImageEvaluationMetric
from .average_meter import AverageMeter
from ..utils import UnsupportedPackage


try:
    from sklearn.metrics import accuracy_score, confusion_matrix
except ImportError as import_error:
    accuracy_score = UnsupportedPackage("sklearn.metric.accuracy_score", import_error.msg)
    confusion_matrix = UnsupportedPackage("sklearn.metric.confusion_matrix", import_error.msg)


def prepare_pha(image):
    if image.shape[-1] == 4:
        return image[:, :, -1].astype(np.float32) / 255
    elif image.shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    elif len(image.shape) == 2:
        return image.astype(np.float32) / 255
    else:
        raise ValueError('Unsupported format of image!')


def prepare_fgr(image):
    if image.shape[-1] == 4:
        return image[:, :, :-1].astype(np.float32) / 255
    elif image.shape[-1] == 3:
        return image.astype(np.float32) / 255
    else:
        raise ValueError('Unsupported format of image!')


class MeanOfAbsoluteDifference(PerImageEvaluationMetric):
    __provider__ = 'mean_of_absolute_difference'
    annotation_types = (BackgroundMattingAnnotation, )
    prediction_types = (BackgroundMattingPrediction, )

    def update(self, annotation, prediction):
        pred = prepare_pha(self.get_prediction(prediction))
        gt = prepare_pha(annotation.value)
        value = np.mean(abs(pred - gt)) * 1e3
        self.results.append(value)
        return value

    def evaluate(self, annotations, predictions):
        return sum(self.results) / len(self.results)

    def get_prediction(self, prediction):
        if self.name.startswith('alpha'):
            return prediction.value['pha']
        else:
            return prediction.value['fgr']

    def reset(self):
        self.results = []

    def configure(self):
        self.reset()

    @classmethod
    def get_common_meta(cls):
        return {'target': 'higher-worse'}


class SpatialGradient(MeanOfAbsoluteDifference):
    __provider__ = 'spatial_gradient'

    def update(self, annotation, prediction):
        pred = prepare_pha(self.get_prediction(prediction))
        gt = prepare_pha(annotation.value)
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
        size = np.int(2 * half_size + 1)
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
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
        self.reset()


class MeanSquaredError(MeanOfAbsoluteDifference):
    __provider__ = 'mean_squared_error'

    def update(self, annotation, prediction):
        pred = prepare_pha(self.get_prediction(prediction))
        gt = prepare_pha(annotation.value)
        value = np.mean((pred - gt) ** 2) * 1e3
        self.results.append(value)
        return value
