"""
Copyright (c) 2018-2020 Intel Corporation

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

import tensorflow as tf
from .launcher import Launcher, LauncherConfigValidator, ListInputsField
from ..config import PathField, StringField

try:
    tf_lite = tf.lite
except AttributeError:
    tf_lite = tf.contrib.lite


class TFLiteLauncher(Launcher):
    __provider__ = 'tf_lite'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'inputs': ListInputsField(optional=True),
            'model': PathField(is_directory=False, description="Path to model."),
            'device': StringField(choices=('cpu', 'gpu'), optional=True, description="Device: cpu or gpu."),
        })
        return parameters

    def __init__(self, config_entry, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        self.default_layout = 'NHWC'
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)

        tf_launcher_config = LauncherConfigValidator(
            'TF_Lite_Launcher', fields=self.parameters(), delayed_model_loading=self._delayed_model_loading
        )
        tf_launcher_config.validate(self.config)
        if not self._delayed_model_loading:
            self._interpreter = tf_lite.Interpreter(model_path=str(self.config['model']))
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._inputs = {input_layer['name']: input_layer for input_layer in self._input_details}
        self.device = '/{}:0'.format(self.config.get('device', 'cpu').lower())

    def predict(self, inputs, metadata=None, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """

        results = []

        for dataset_input in inputs:
            self.set_tensors(dataset_input)
            self._interpreter.invoke()
            res = {output['name']: self._interpreter.get_tensor(output['index']) for output in self._output_details}
            results.append(res)

            if metadata is not None:
                for meta_ in metadata:
                    meta_['input_shape'] = self.inputs_info_for_meta()

        return results

    @property
    def batch(self):
        return 1

    @property
    def inputs(self):
        return self._inputs

    def release(self):
        del self._interpreter

    def predict_async(self, *args, **kwargs):
        raise ValueError('TensorFlow Lite Launcher does not support async mode yet')

    @property
    def output_blob(self):
        return next(iter(self._output_details))['name']

    def set_tensors(self, dataset_input):
        """
        Set input tensors:
        :param dataset_input: dict {"input_layer_name": input_data}
        :return: None
        """
        for layer, data in dataset_input.items():
            self._interpreter.set_tensor(
                self._inputs[layer]['index'], data.astype(self._inputs[layer]['dtype'])
            )
