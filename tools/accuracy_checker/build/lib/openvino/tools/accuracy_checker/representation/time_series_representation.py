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

from .base_representation import BaseRepresentation


class TimeSeriesRepresentation(BaseRepresentation):
    pass


class TimeSeriesForecastingAnnotation(TimeSeriesRepresentation):
    def __init__(self, identifier, inputs, outputs, mean, scale):
        super().__init__(identifier)
        self.inputs = inputs
        self.outputs = outputs
        self.mean = mean
        self.scale = scale

    def inorm(self, var):
        return var * self.scale + self.mean


class TimeSeriesForecastingQuantilesPrediction(TimeSeriesRepresentation):
    def __init__(self, identifier):
        super().__init__(identifier)
        self.preds = {}

    def __setitem__(self, quantile, value):
        self.preds[quantile] = value

    def __getitem__(self, quantile):
        return self.preds[quantile]
