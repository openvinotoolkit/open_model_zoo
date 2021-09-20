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
from ..representation import TimeSeriesForecastingQuantilesPrediction, TimeSeriesForecastingAnnotation


class TimeSeriesDenormalize(Postprocessor):
    __provider__ = 'time_series_denormalize'
    annotation_types = (TimeSeriesForecastingAnnotation, )
    prediction_types = (TimeSeriesForecastingQuantilesPrediction, )

    def process_image(self, annotation, prediction):
        for i, _ in enumerate(annotation):
            annotation[i].outputs = annotation[i].inorm(annotation[i].outputs)
            for k in prediction[i].preds.keys():
                prediction[i].preds[k] = annotation[i].inorm(prediction[i].preds[k])
        return annotation, prediction
