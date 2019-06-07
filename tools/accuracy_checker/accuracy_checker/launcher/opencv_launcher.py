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
from collections import OrderedDict
import numpy as np
import cv2

from ..config import PathField, StringField, ConfigError, ListInputsField
from ..logging import print_info
from .launcher import Launcher, LauncherConfigValidator
from ..utils import get_or_parse_value

DEVICE_REGEX = r'(?P<device>cpu$|gpu|gpu_fp16)?'
BACKEND_REGEX = r'(?P<backend>ocv|ie)?'


class OpenCVLauncherConfigValidator(LauncherConfigValidator):
    def validate(self, entry, field_uri=None):
        super().validate(entry, field_uri)
        inputs = entry.get('inputs')
        for input_layer in inputs:
            if 'shape' not in input_layer:
                raise ConfigError('input value should have shape field')


class OpenCVLauncher(Launcher):
    """
    Class for infer model using OpenCV library.
    """
    __provider__ = 'opencv'

    OPENCV_BACKENDS = {
        'ocv': cv2.dnn.DNN_BACKEND_OPENCV,
        'ie': cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE
    }

    TARGET_DEVICES = {
        'cpu': cv2.dnn.DNN_TARGET_CPU,
        'gpu': cv2.dnn.DNN_TARGET_OPENCL,
        'gpu_fp16': cv2.dnn.DNN_TARGET_OPENCL_FP16
    }

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model file."),
            'weights': PathField(description="Path to weights file.", optional=True, default='', check_exists=False),
            'device': StringField(
                regex=DEVICE_REGEX, choices=OpenCVLauncher.TARGET_DEVICES.keys(),
                description="Device name: {}".format(', '.join(OpenCVLauncher.TARGET_DEVICES.keys()))
            ),
            'backend': StringField(
                regex=BACKEND_REGEX, choices=OpenCVLauncher.OPENCV_BACKENDS.keys(),
                optional=True, default='IE',
                description="Backend name: {}".format(', '.join(OpenCVLauncher.OPENCV_BACKENDS.keys()))),
            'inputs': ListInputsField(optional=False, description="Inputs.")
        })

        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)

        opencv_launcher_config = OpenCVLauncherConfigValidator('OpenCV_Launcher', fields=self.parameters())
        opencv_launcher_config.validate(self.config)

        self.model = str(self.get_value_from_config('model'))
        self.weights = str(self.get_value_from_config('weights'))

        self.network = cv2.dnn.readNet(self.model, self.weights)

        match = re.match(BACKEND_REGEX, self.get_value_from_config('backend').lower())
        selected_backend = match.group('backend')
        print_info('backend: {}'.format(selected_backend))
        backend = OpenCVLauncher.OPENCV_BACKENDS.get(selected_backend)

        self.network.setPreferableBackend(backend)

        match = re.match(DEVICE_REGEX, self.get_value_from_config('device').lower())
        selected_device = match.group('device')

        if 'tags' in self.config:
            tags = self.config['tags']
            if ('FP16' in tags) and (selected_device == 'gpu'):
                selected_device = 'gpu_fp16'

        target = OpenCVLauncher.TARGET_DEVICES.get(selected_device)

        if target is None:
            raise ConfigError('{} is not supported device'.format(selected_device))

        self.network.setPreferableTarget(target)

        inputs = self.config['inputs']

        def parse_shape_value(shape):
            return tuple([1, *[int(elem) for elem in get_or_parse_value(shape, ())]])

        self._inputs_shapes = OrderedDict({elem.get('name'): parse_shape_value(elem.get('shape')) for elem in inputs})
        self.network.setInputsNames(list(self._inputs_shapes.keys()))
        self.output_names = self.network.getUnconnectedOutLayersNames()

    @property
    def inputs(self):
        """
        Returns:
            inputs in NCHW format.
        """
        return self._inputs_shapes

    @property
    def batch(self):
        return 1

    @property
    def output_blob(self):
        return next(iter(self.output_names))

    def predict(self, inputs, metadata, *args, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """
        results = []
        for input_blobs in inputs:
            for blob_name in self._inputs_shapes:
                self.network.setInput(input_blobs[blob_name].astype(np.float32), blob_name)
            list_prediction = self.network.forward(self.output_names)
            dict_result = {
                output_name: output_value for output_name, output_value in zip(self.output_names, list_prediction)
            }
            results.append(dict_result)

        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError('OpenCV Launcher does not support async mode yet')

    def release(self):
        """
        Releases launcher.
        """
        del self.network
