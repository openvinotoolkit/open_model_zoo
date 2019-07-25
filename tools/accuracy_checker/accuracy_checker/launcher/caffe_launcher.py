"""
Copyright (c) 2019 Intel Corporation

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

import caffe
import numpy as np

from ..config import PathField, StringField, NumberField, BoolField
from .launcher import Launcher, LauncherConfigValidator

DEVICE_REGEX = r'(?P<device>cpu$|gpu)(_(?P<identifier>\d+))?'


class CaffeLauncher(Launcher):
    """
    Class for infer model using Caffe framework.
    """

    __provider__ = 'caffe'

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)

        caffe_launcher_config = LauncherConfigValidator('Caffe_Launcher', fields=self.parameters())
        caffe_launcher_config.validate(self.config)

        self.model = str(self.get_value_from_config('model'))
        self.weights = str(self.get_value_from_config('weights'))

        self.network = caffe.Net(self.model, self.weights, caffe.TEST)
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
            'model': PathField(description="Path to model."),
            'weights': PathField(description="Path to model."),
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

    def fit_to_input(self, data, layer_name, layout):
        data_shape = np.shape(data)
        data = np.transpose(data, layout) if len(data_shape) == 4 else np.array(data)
        layer_shape = self.inputs[layer_name]
        if layer_shape != data_shape:
            self.network.blobs[layer_name].reshape(*data.shape)

        return data

    def predict(self, inputs, metadata, *args, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """
        results = []
        for infer_input in inputs:
            results.append(self.network.forward(**infer_input))
            for image_meta in metadata:
                image_meta['input_shape'] = self.inputs_info_for_meta()

        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError('Caffe Launcher does not support async mode')

    def release(self):
        """
        Releases launcher.
        """
        del self.network
