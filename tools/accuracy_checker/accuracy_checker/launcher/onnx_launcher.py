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
import numpy as np
import onnxruntime.backend as backend
from ..config import PathField, StringField
from .launcher import Launcher, LauncherConfigValidator

DEVICE_REGEX = r'(?P<device>cpu$|gpu)'


class ONNXLauncher(Launcher):
    __provider__ = 'onnx_runtime'

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)

        onnx_launcher_config = LauncherConfigValidator('ONNX_Launcher', fields=self.parameters())
        onnx_launcher_config.validate(self.config)

        self.model = str(self.get_value_from_config('model'))

        device = re.match(DEVICE_REGEX, self.get_value_from_config('device').lower()).group('device')
        beckend_rep = backend.prepare(model=self.model, device=device.upper())
        self._inference_session = beckend_rep._session # pylint: disable=W0212
        outputs = self._inference_session.get_outputs()
        self.output_names = [output.name for output in outputs]

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model."),
            'device': StringField(regex=DEVICE_REGEX, description="Device name.", optional=True, default='CPU'),
        })

        return parameters

    @property
    def inputs(self):
        inputs_info = self._inference_session.get_inputs()
        return {input_layer.name: input_layer.shape for input_layer in inputs_info}

    @property
    def output_blob(self):
        return next(iter(self.output_names))

    @property
    def batch(self):
        return 1

    def predict(self, inputs, metadata, *args, **kwargs):
        results = []
        for infer_input in inputs:
            prediction_list = self._inference_session.run(self.output_names, infer_input)
            results.append(
                {output_name: prediction for output_name, prediction in zip(self.output_names, prediction_list)}
            )
            for meta_ in metadata:
                meta_['input_shape'] = self.inputs_info_for_meta()

        return results

    @staticmethod
    def fit_to_input(data, layer_name, layout):
        if len(np.shape(data)) == 4:
            return np.transpose(data, layout).astype(np.float32)
        return np.array(data).astype(np.float32)

    def predict_async(self, *args, **kwargs):
        raise ValueError('ONNX Runtime Launcher does not support async mode yet')

    def release(self):
        del self._inference_session
