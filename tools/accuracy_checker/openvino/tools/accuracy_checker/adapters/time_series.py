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

from .adapter import Adapter
from ..representation import TimeSeriesForecastingQuantilesPrediction
from ..config import StringField, DictField


class QuantilesPredictorAdapter(Adapter):
    __provider__ = 'quantiles_predictor'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'quantiles': DictField(
                allow_empty=False,
                description="preds[i]->quantile[i] mapping."
            ),
            "output_name": StringField()
        })
        return parameters

    def configure(self):
        self.quantiles = self.get_value_from_config('quantiles')
        self.output_name = str(self.get_value_from_config('output_name'))

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        output = raw_outputs[self.output_name]
        preds = TimeSeriesForecastingQuantilesPrediction(identifiers[0])
        for k, v in self.quantiles.items():
            preds[k] = output[:, :, v]
        return [preds]
