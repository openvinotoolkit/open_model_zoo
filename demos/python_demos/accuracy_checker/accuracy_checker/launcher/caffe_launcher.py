"""
 Copyright (c) 2018 Intel Corporation

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

from ..utils import parse_inputs, check_user_inputs
from ..config import NumberField
from ..config import PathField, StringField
from .launcher import Launcher, LauncherConfig

DEVICE_REGEX = r'(?P<device>cpu$|gpu)(_(?P<identifier>\d+))?'


class CaffeLauncherConfig(LauncherConfig):
    """
    Specifies configuration structure for Caffe launcher
    """
    model = PathField(check_exists=True, is_directory=False)
    weights = PathField(check_exists=True, is_directory=False)
    device = StringField(regex=DEVICE_REGEX)
    batch = NumberField(floats=False, min_value=1, optional=True)
    output_name = StringField(optional=True)


class CaffeLauncher(Launcher):
    """
    Class for infer model using Caffe framework
    """
    __provider__ = 'caffe'

    def __init__(self, config_entry: dict, adapter, *args, **kwargs):
        super().__init__(config_entry, adapter, *args, **kwargs)

        caffe_launcher_config = CaffeLauncherConfig('Caffe_Launcher')
        caffe_launcher_config.validate(self._config)

        self.model = self._config['model']
        self.weights = self._config['weights']

        self.network = caffe.Net(self.model, self.weights, caffe.TEST)

        match = re.match(DEVICE_REGEX, self._config['device'].lower())
        if match.group('device') == 'gpu':
            caffe.set_mode_gpu()
            identifier = match.group('identifier')
            if identifier is None:
                identifier = 0
            caffe.set_device(int(identifier))
        elif match.group('device') == 'cpu':
            caffe.set_mode_cpu()

        self._batch = self._config.get('batch', 1)

        self._config_inputs = parse_inputs(self._config.get('inputs', []))
        check_user_inputs(self.network.inputs, self._config_inputs)

        self._inputs_shapes = {}
        for input_blob in self.network.inputs:
            if input_blob in self._config_inputs:
                continue

            channels, height, width = self.network.blobs[input_blob].data.shape[1:]
            self.network.blobs[input_blob].reshape(self._batch, channels, height, width)
            self._inputs_shapes[input_blob] = channels, height, width

            self.adapter.output_blob = self.adapter.output_blob or next(iter(self.network.outputs))

    @property
    def inputs(self):
        """
        Returns:
            inputs in NCHW format
        """
        return self._inputs_shapes

    @property
    def batch(self):
        return self._batch

    def predict(self, identifiers, data, *args, **kwargs):
        """
        Args:
            identifiers: list of input data identifiers
            data: input data
        Returns:
            output of model converted to appropriate representation
        """
        data = np.transpose(data, (0, 3, 1, 2))
        dataset_inputs = {}
        for input_blob in self.network.inputs:
            if input_blob in self._config_inputs:
                continue

            if data.shape[0] != self._batch:
                self.network.blobs[input_blob].reshape(data.shape[0], *self.network.blobs[input_blob].data.shape[1:])

            dataset_inputs[input_blob] = data

        res = self.network.forward(**self._config_inputs, **dataset_inputs)

        if self.adapter is not None:
            res = self.adapter(res, identifiers)

        return res

    def release(self):
        """
        Releases launcher
        """
        del self.network
