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

from ..adapters import Adapter
from ..config import ConfigValidator, DictField
from ..representation import ClassificationPrediction, ContainerPrediction


class AttributeClassificationAdapter(Adapter):
    """
    Class for converting output of attributes classification model to
    multiple ClassificationPrediction representations which are contained
    in ContainerPrediction representation
    """
    __provider__ = 'attribute_classification'
    prediction_types = (ClassificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'output_layer_map': DictField(
                allow_empty=False,
                description="Output layer in map (Key: output layer name, Value: attribute)"
            )
        })
        return parameters

    def configure(self):
        super().configure()
        self.output_layers = self.get_value_from_config('output_layer_map')
        self.outputs_verified = False

    @classmethod
    def validate_config(cls, config, fetch_only=False, **kwargs):
        return super().validate_config(
            config, fetch_only=fetch_only, on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT
        )

    def select_output_blob(self, outputs):
        new_output_layers = {}
        for attr, layer_name in self.output_layers.items():
            new_output_layers[attr] = self.check_output_name(layer_name, outputs)
        self.output_layers = new_output_layers
        self.outputs_verified = True

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: list of meta information about each frame
        Returns:
            list of ContainerPrediction objects
        """
        result = []
        if isinstance(raw, dict):
            raw = [raw]
        if not self.outputs_verified:
            self.select_output_blob(raw)
        for identifier, raw_output in zip(identifiers, raw):
            container_dict = {}
            for layer_name, attribute in self.output_layers.items():
                container_dict[attribute] = ClassificationPrediction(
                    identifier, raw_output[layer_name].reshape(-1))
            result.append(ContainerPrediction(container_dict))
        return result
