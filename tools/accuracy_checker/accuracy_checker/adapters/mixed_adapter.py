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

from .adapter import Adapter, create_adapter
from ..config import DictField
from ..representation import ContainerPrediction


class MixedAdapter(Adapter):
    __provider__ = 'mixed'

    # this will be set after reading adapter configs
    prediction_types = ()

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'adapters': DictField(
                allow_empty=False,
                description='Dict where key is output name and value is adapter config map including'
                            'key "output_blob" to indicating output layer of model')
        })
        return parameters

    def __create_adapter(self, adapter_config, output_blob):
        adapter = create_adapter(adapter_config)
        adapter.launcher_config = self.launcher_config
        adapter.output_blob = output_blob
        return adapter

    def configure(self):
        adapters = self.get_value_from_config('adapters')
        self.adapters = {}
        for output_name, adapter_config in adapters.items():
            layer_name = adapter_config.pop('output_blob')
            self.adapters[layer_name] = (output_name, self.__create_adapter(adapter_config, layer_name))
        prediction_types = set()
        for _, adapter in self.adapters.values():
            prediction_types.update(adapter.prediction_types)
        self.prediction_types = tuple(prediction_types)

    @staticmethod
    def is_result_valid(result: dict):
        '''
        this method check whether values of result dict have the same length
        '''
        list_len = -1
        for val in result.values():
            if list_len < 0:
                list_len = len(val)
            else:
                if list_len != len(val):
                    return False
        return True

    def process(self, raw, identifiers, frame_meta):
        result = {}

        for layer, (_, adapter) in self.adapters.items():
            result[layer] = adapter.process(raw, identifiers, frame_meta)

        if not self.is_result_valid(result):
            raise RuntimeError("length of predictions from each adapter should be same")

        output = []

        for i, _ in enumerate(identifiers):
            container_args = {}
            for layer, (name, _) in self.adapters.items():
                if isinstance(result[layer][i], ContainerPrediction):
                    container_args.update(result[layer][i].representations)
                else:
                    container_args[name] = result[layer][i]
            output.append(ContainerPrediction(container_args))

        return output
