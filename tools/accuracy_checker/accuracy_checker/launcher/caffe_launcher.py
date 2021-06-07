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

import re
from pathlib import Path

import caffe
import numpy as np

from ..config import PathField, StringField, NumberField, BoolField, ConfigError
from .launcher import Launcher
from ..logging import print_info

DEVICE_REGEX = r'(?P<device>cpu$|gpu)(_(?P<identifier>\d+))?'


class CaffeLauncher(Launcher):
    """
    Class for infer model using Caffe framework.
    """

    __provider__ = 'caffe'

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)

        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)
        self.validate_config(config_entry, delayed_model_loading=self._delayed_model_loading)
        self._do_reshape = False

        if not self._delayed_model_loading:
            self.model, self.weights = self.automatic_model_search()
            self.network = caffe.Net(str(self.model), str(self.weights), caffe.TEST)
        self.allow_reshape_input = self.get_value_from_config('allow_reshape_input')

        match = re.match(DEVICE_REGEX, self.get_value_from_config('device').lower())
        if match.group('device') == 'gpu':
            caffe.set_mode_gpu()
            identifier = match.group('identifier') or 0
            caffe.set_device(int(identifier))
        elif match.group('device') == 'cpu':
            caffe.set_mode_cpu()
        self._batch = self.get_value_from_config('batch')

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model.", file_or_directory=True),
            'weights': PathField(description="Path to weights.", optional=True, file_or_directory=True),
            'device': StringField(regex=DEVICE_REGEX, description="Device name."),
            'batch': NumberField(
                value_type=int, min_value=1, optional=True, default=1, description="Batch size."
            ),
            'allow_reshape_input': BoolField(optional=True, default=False, description="Allows reshape input.")
        })

        return parameters

    @property
    def inputs(self):
        """
        Returns:
            inputs in NCHW format.
        """
        inputs_map = {}
        for input_blob in self.network.inputs:
            inputs_map[input_blob] = self.network.blobs[input_blob].data.shape

        return inputs_map

    @property
    def batch(self):
        return self._batch

    @property
    def output_blob(self):
        return next(iter(self.network.outputs))

    def fit_to_input(self, data, layer_name, layout, precision):
        data_shape = np.shape(data)
        layer_shape = self.inputs[layer_name]
        if len(data_shape) == 5 and len(layer_shape) == 4:
            data = data[0]
            data_shape = np.shape(data)
        data = np.transpose(data, layout) if len(data_shape) == 4 and layout is not None else np.array(data)
        data_shape = np.shape(data)
        if layer_shape != data_shape:
            self._do_reshape = True

        return data.astype(precision) if precision else data

    def automatic_model_search(self):
        model = Path(self.get_value_from_config('model'))
        weights = self.get_value_from_config('weights')
        if model.is_dir():
            models_list = list(model.glob('{}.prototxt'.format(self._model_name)))
            if not models_list:
                models_list = list(model.glob('*.prototxt'))
            if not models_list:
                raise ConfigError('Suitable model description is not detected')
            if len(models_list) != 1:
                raise ConfigError('Several suitable models found, please specify required model')
            model = models_list[0]
        if weights is None or Path(weights).is_dir():
            weights_dir = weights or model.parent
            weights = Path(weights_dir) / model.name.replace('prototxt', 'caffemodel')
            if not weights.exists():
                weights_list = list(Path(weights_dir).glob('*.caffemodel'))
                if not weights_list:
                    raise ConfigError('Suitable weights is not detected')
                if len(weights_list) != 1:
                    raise ConfigError('Several suitable weights found, please specify required explicitly')
                weights = weights_list[0]
        accepted_suffixes = ['.prototxt']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('Found model {}'.format(model))
        weights = Path(weights)
        accepted_weights_suffixes = ['.caffemodel']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('Found weights {}'.format(weights))

        return model, weights

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
            if self._do_reshape:
                for layer_name, data in infer_input.items():
                    if data.shape != self.inputs[layer_name]:
                        self.network.blobs[layer_name].reshape(*data.shape)

            results.append(self.network.forward(**infer_input))
        if metadata is not None:
            for image_meta in metadata:
                image_meta['input_shape'] = self.inputs_info_for_meta()

        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError('Caffe Launcher does not support async mode')

    @staticmethod
    def create_network(model, weights):
        return caffe.Net(str(model), str(weights), caffe.TEST)

    def release(self):
        """
        Releases launcher.
        """
        del self.network
