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
import numpy as np
import onnxruntime.backend as backend
import onnxruntime as onnx_rt
from ..logging import warning
from ..config import PathField, StringField, ListField, ConfigError
from .launcher import Launcher
from ..utils import contains_all
from ..logging import print_info


DEVICE_REGEX = r'(?P<device>cpu$|gpu)'


class ONNXLauncher(Launcher):
    __provider__ = 'onnx_runtime'

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)

        self.validate_config(config_entry, delayed_model_loading=self._delayed_model_loading)
        if not self._delayed_model_loading:
            self.model = self.automatic_model_search()
            self._inference_session = self.create_inference_session(str(self.model))
            outputs = self._inference_session.get_outputs()
            self.output_names = [output.name for output in outputs]
            self._input_precisions = {}
            for input_info in self._inference_session.get_inputs():
                dtype = input_info.type.replace('tensor(', '').replace(')', '')
                if dtype == 'float':
                    dtype += '32'
                self._input_precisions[input_info.name] = dtype

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model.", file_or_directory=True),
            'device': StringField(regex=DEVICE_REGEX, description="Device name.", optional=True, default='CPU'),
            'execution_providers': ListField(
                value_type=StringField(description="Execution provider name.", ),
                default=['CPUExecutionProvider'], optional=True
            )
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

    def automatic_model_search(self):
        model = Path(self.get_value_from_config('model'))
        if model.is_dir():
            model_list = list(model.glob('{}.onnx'.format(self._model_name)))
            if not model_list:
                model_list = list(model.glob('*.onnx'))
                if not model_list:
                    raise ConfigError('Model not found')
            if len(model_list) != 1:
                raise ConfigError('Several suitable models found, please specify explicitly')
            model = model_list[0]
        accepted_suffixes = ['.onnx']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('Found model {}'.format(model))

        return model

    def create_inference_session(self, model):
        if 'execution_providers' in self.config:
            try:
                session = self._create_session_via_execution_providers_api(model)
                return session
            except AttributeError:
                warning('Execution Providers API is not supported, onnxruntime switched on Backend API')
        return self._create_session_via_backend_api(model)

    def _create_session_via_execution_providers_api(self, model):
        session_options = onnx_rt.SessionOptions()
        session = onnx_rt.InferenceSession(model, sess_options=session_options)
        self.execution_providers = self.get_value_from_config('execution_providers')
        available_providers = session.get_providers()
        contains_all(available_providers, self.execution_providers)
        session.set_providers(self.execution_providers)

        return session

    def _create_session_via_backend_api(self, model):
        self.device = re.match(DEVICE_REGEX, self.get_value_from_config('device').lower()).group('device')
        beckend_rep = backend.prepare(model=str(model), device=self.device.upper())
        return beckend_rep._session  # pylint: disable=W0212

    def predict(self, inputs, metadata=None, **kwargs):
        results = []
        for infer_input in inputs:
            prediction_list = self._inference_session.run(self.output_names, infer_input)
            results.append(dict(zip(self.output_names, prediction_list)))
            if metadata is not None:
                for meta_ in metadata:
                    meta_['input_shape'] = self.inputs_info_for_meta()

        return results

    def fit_to_input(self, data, layer_name, layout, precision):
        layer_shape = self.inputs[layer_name]
        input_precision = self._input_precisions.get(layer_name, np.float32) if not precision else precision
        if len(np.shape(data)) == 4:
            if layout:
                data = np.transpose(data, layout).astype(input_precision)
            if len(layer_shape) == 3:
                if np.shape(data)[0] != 1:
                    raise ValueError('Only for batch size 1 first dimension can be omitted')
                return data[0].astype(input_precision)
            return data.astype(input_precision)
        if len(np.shape(data)) == 5 and len(layout) == 5:
            return np.transpose(data, layout).astype(input_precision)
        if len(np.shape(data))-1 == len(layer_shape):
            return np.array(data[0]).astype(input_precision)
        return np.array(data).astype(input_precision)

    def predict_async(self, *args, **kwargs):
        raise ValueError('ONNX Runtime Launcher does not support async mode yet')

    def release(self):
        if hasattr(self, '_inference_session'):
            del self._inference_session
