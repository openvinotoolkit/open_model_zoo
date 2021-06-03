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
from .launcher import Launcher
from ..config import BaseField, ListField, PathField, StringField, ConfigError


class TF2Launcher(Launcher):
    __provider__ = 'tf2'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'saved_model_dir': PathField(is_directory=True, optional=True, description='Path to saved model directory'),
            'device': StringField(
                choices=('cpu', 'gpu'), default='cpu', optional=True, description="Device name: cpu or gpu"),
            'inputs': BaseField(optional=True, description="Inputs."),
            'output_names': ListField(
                allow_empty=False, optional=True, value_type=StringField(), description="Output names."
            )
        })
        return parameters

    def __init__(self, config_entry, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        try:
            import tensorflow # pylint: disable=C0415
            if tensorflow.__version__ < '2.0.0':
                raise ConfigError('TF2 launcher supports only tensorflow >= 2.0')
        except ImportError as import_error:
            raise ValueError(
                "TensorFlow isn't installed. Please, install it before using. \n{}".format(import_error.msg)
            )
        self.tf = tensorflow
        self.default_layout = 'NHWC'
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)
        self.validate_config(config_entry, delayed_model_loading=self._delayed_model_loading)

        if not self._delayed_model_loading:
            if 'saved_model_dir' not in self.config:
                raise ConfigError('saved model directory should be provided')

            self._config_outputs = self.get_value_from_config('output_names')
            self._model_fn = self._load_saved_model(str(self.get_value_from_config('saved_model_dir')))
        self.device = '/{}:0'.format(self.get_value_from_config('device').lower())

    def predict(self, inputs, metadata=None, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """
        results = []
        for infer_input in inputs:
            outputs = self._model_fn(**infer_input)
            res = {key: value.numpy() for key, value in outputs.items()}
            results.append(res)
            if metadata is not None:
                for meta_ in metadata:
                    meta_['input_shape'] = self.inputs_info_for_meta(infer_input)

        return results

    def fit_to_input(self, data, layer_name, layout, precision):
        if layout is not None and len(np.shape(data)) == len(layout):
            data = np.transpose(data, layout)
        else:
            data = np.array(data)
        precision = self.tf.as_dtype(precision) if precision else self.inputs[layer_name]['precision']
        return self.tf.convert_to_tensor(data, dtype=precision)

    def inputs_info_for_meta(self, feed_dict=None):
        if feed_dict is None:
            return super().inputs_info_for_meta()
        return {
            input_name: tuple(input_data.shape)
            for input_name, input_data in feed_dict.items()
        }

    @property
    def batch(self):
        return 1

    @property
    def inputs(self):
        graph_inputs = self._get_inputs()
        return {
            node_name.split('import/')[-1]: {
                'shape': tuple(node.shape),
                'precision': self.tf.as_dtype(node.dtype)
            }
            for node_name, node in graph_inputs.items()
        }

    def release(self):
        del self._model_fn
        if hasattr(self, '_loaded'):
            del self._loaded

    @property
    def output_blob(self):
        return next(iter(self._model_fn.structured_outputs))

    def predict_async(self, *args, **kwargs):
        raise ValueError('TensorFlow Launcher does not support async mode yet')

    def _load_saved_model(self, model_dir):
        self._loaded = self.tf.saved_model.load(model_dir)
        self._model_fn = self._loaded.signatures["serving_default"]
        return self._model_fn

    def _get_inputs(self):
        inputs = [node for node in self._model_fn.inputs if node.name.split(':')[0] in self._model_fn._arg_keywords]

        return {node.name.split(':')[0]: node for node in inputs}
