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

from .postprocessor import Postprocessor
from ..representation import ElectricityTimeSeriesForecastingPrediction, ElectricityTimeSeriesForecastingAnnotation


class ElectricityTimeSeriesDenormalize(Postprocessor):
    __provider__ = 'electricity_time_series_denormalize'
    annotation_types = (ElectricityTimeSeriesForecastingAnnotation, )
    prediction_types = (ElectricityTimeSeriesForecastingPrediction, )

    def process_image(self, annotation, prediction):
        for i in range(len(annotation)):
            mean, scale = annotation[i].mean, annotation[i].scale
            annotation[i].outputs = self._denorm(annotation[i].outputs, mean, scale)
            for k in prediction[i].preds.keys():
                prediction[i].preds[k] = self._denorm(prediction[i].preds[k], mean, scale)
        return annotation, prediction

    def _denorm(self, x, mean, scale):
        return x * scale + mean
