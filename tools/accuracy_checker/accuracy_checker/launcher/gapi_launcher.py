"""
Copyright (c) 2018-2024 Intel Corporation

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

from pathlib import Path
import os
import numpy as np

import cv2

from ..config import PathField, StringField, ConfigError, ListInputsField, ListField, BoolField
from ..logging import print_info, warning
from .launcher import Launcher, LauncherConfigValidator
from .dlsdk_launcher_config import MULTI_DEVICE_KEYWORD, HETERO_KEYWORD, NIREQ_REGEX
from ..utils import get_or_parse_value, get_path

try:
    compile_args = cv2.compile_args
except AttributeError:
    try:
        compile_args = cv2.gapi.compile_args
    except AttributeError:
        def compile_args(*args):
            return list(map(cv2.GCompileArg, args))

try:
    from openvino.inference_engine import IECore  # pylint: disable=import-outside-toplevel,package-absolute-imports
    _ie_core = IECore()
except ImportError:
    _ie_core = None


try:
    from openvino.inference_engine import known_plugins  # pylint: disable=import-outside-toplevel,package-absolute-imports
except ImportError:
    known_plugins = []


class GAPILauncherConfigValidator(LauncherConfigValidator):
    def create_device_regex(self, available_devices):
        self.regular_device_regex = r"(?:^(?P<device>{devices})$)".format(devices="|".join(available_devices))
        self.hetero_regex = r"(?:^{hetero}(?P<devices>(?:{devices})(?:,(?:{devices}))*)$)".format(
            hetero=HETERO_KEYWORD, devices="|".join(available_devices)
        )
        self.multi_device_regex = r"(?:^{multi}(?P<devices_ireq>(?:{devices_ireq})(?:,(?:{devices_ireq}))*)$)".format(
            multi=MULTI_DEVICE_KEYWORD, devices_ireq="{}?|".format(NIREQ_REGEX).join(available_devices)
        )
        self.supported_device_regex = r"{multi}|{hetero}|{regular}".format(
            multi=self.multi_device_regex, hetero=self.hetero_regex, regular=self.regular_device_regex
        )
        self.fields['device'].set_regex(self.supported_device_regex)

    def validate(self, entry, field_uri=None, ie_core=None, fetch_only=False, validation_scheme=None):
        self.fields['inputs'].optional = self.delayed_model_loading
        error_stack = super().validate(entry, field_uri)
        if not self.delayed_model_loading:
            inputs = entry.get('inputs')
            for input_layer in inputs:
                if 'shape' not in input_layer:
                    if not fetch_only:
                        raise ConfigError('input value should have shape field')
                    error_stack.extend(self.build_error(entry, field_uri, 'input value should have shape field'))
        self.create_device_regex(known_plugins)
        if 'device' not in entry or entry.get('backend', 'ie') == 'mx':
            return error_stack
        try:
            self.fields['device'].validate(
                entry['device'], field_uri, validation_scheme=(validation_scheme or {}).get('device')
            )
        except ConfigError as error:
            if ie_core is not None:
                self.create_device_regex(ie_core.available_devices)
                try:
                    self.fields['device'].validate(
                        entry['device'], field_uri, validation_scheme=(validation_scheme or {}).get('device')
                    )
                except ConfigError:
                    # workaround for devices where this metric is non implemented
                    warning('unknown device: {}'.format(entry['device']))
            else:
                if not fetch_only:
                    raise error
                error_stack.append(error)
        return error_stack


class GAPILauncher(Launcher):
    __provider__ = 'g-api'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model file.", file_or_directory=True),
            'weights': PathField(description="Path to weights file.",
                                 optional=True, file_or_directory=True),
            'backend': StringField(optional=True, description='backend name', choices=['ie', 'mx'], default='ie'),
            'device': StringField(description="Device name", optional=True, default='cpu'),
            '_model_is_blob': BoolField(optional=True, description='hint for auto model search'),
            'inputs': ListInputsField(optional=False, description="Inputs."),
            'outputs': ListField(value_type=str, allow_empty=False, description='Outputs.')
        })

        return parameters

    def __init__(self, config_entry: dict, *args, preprocessor=None, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)
        self.validate_config(config_entry, delayed_model_loading=self._delayed_model_loading)
        self.backend = self.get_value_from_config('backend')
        print_info('backend: {}'.format(self.backend))
        self.device = self.get_value_from_config('device').upper() if self.backend == 'ie' else ''
        self.comp = None
        self.network_args = None
        self.output_names = self.get_value_from_config('outputs')
        self._inputs_shapes, self._const_inputs = self.get_inputs_from_config(self.config)
        multi_input = (len(self._inputs_shapes) - len(self.const_inputs)) > 1
        self.non_image_inputs = False
        self.preprocessor = preprocessor
        if multi_input:
            for name, shape in self._inputs_shapes.items():
                if name in self.const_inputs:
                    continue
                if len(shape) != 4:
                    self.non_image_inputs = True
                    break
                if shape[1] not in [1, 3, 4]:
                    self.non_image_inputs = True
        if self.non_image_inputs:
            nhwc_shapes = {}
            for input_name, input_shape in self._inputs_shapes.items():
                if len(input_shape) == 4 and input_shape[1] in [1, 3, 4]:
                    nhwc_shapes[input_name] = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]
                self._inputs_shapes.update(nhwc_shapes)
            self.default_layout = 'NHWC'

        if not self._delayed_model_loading:
            self.model, self.weights = self.automatic_model_search()
            self.prepare_net()

    @classmethod
    def validate_config(cls, config, delayed_model_loading=False, fetch_only=False, uri_prefix=''):
        return GAPILauncherConfigValidator(
            uri_prefix or 'launcher.{}'.format(cls.__provider__),
            fields=cls.parameters(), delayed_model_loading=delayed_model_loading
        ).validate(config, ie_core=_ie_core, fetch_only=fetch_only)

    def prepare_net(self):
        inputs = cv2.GInferInputs()
        g_inputs = []
        for input_name in self.inputs:
            if input_name in self.const_inputs:
                continue
            g_in = cv2.GMat()
            inputs.setInput(input_name, g_in)
            g_inputs.append(g_in)

        outputs = cv2.gapi.infer("net", inputs)
        g_outputs = [outputs.at(out_name) for out_name in self.output_names]
        self.comp = cv2.GComputation(cv2.GIn(*g_inputs), cv2.GOut(*g_outputs))
        args = ['net', str(self.model)]
        if self.weights is not None:
            args.append(str(self.weights))
        args.append(self.device.upper())
        if self.backend == 'ie':
            pp = cv2.gapi.ie.params(*args)
        else:
            pp = cv2.gapi.mx.params('net', str(self.model))
            if self.preprocessor:
                steps = [step.value for step in self.preprocessor.ie_preprocess_steps]
                if steps:
                    pp.cfgPreprocList(steps)
        for input_name, value in self._const_inputs.items():
            pp.constInput(input_name, value)
        if self.backend == 'ie':
            self.network_args = compile_args(cv2.gapi.networks(pp))
        else:
            mvcmd_file = os.environ.get('MVCMD_FILE', '')
            self.network_args = compile_args(
                cv2.gapi.networks(pp), cv2.gapi_mx_mvcmdFile(mvcmd_file)
            )

    @property
    def inputs(self):
        return self._inputs_shapes

    @property
    def batch(self):
        return 1

    @property
    def output_blob(self):
        return next(iter(self.output_names))

    def fit_to_input(self, data, layer_name, layout, precision, template=None):
        if self.non_image_inputs:
            return self._fit_to_input(data, layer_name, layout, precision)
        if np.ndim(data) == 4:
            data = data[0]
        else:
            data = np.array(data)
        if data.dtype in [float, np.float64] and precision is None:
            data = data.astype(np.float32)
        if precision:
            data = data.astype(precision)

        return data

    def _fit_to_input(self, data, layer_name, layout, precision):
        layer_shape = self.inputs[layer_name]
        data = self._data_to_blob(layer_shape, data, layout)
        if precision:
            data = data.astype(precision)
        return self._align_data_shape(data, layer_name)

    @staticmethod
    def _data_to_blob(layer_shape, data, layout): # pylint:disable=R0911
        data_shape = np.shape(data)
        if len(layer_shape) == 4:
            if len(data_shape) == 5:
                data = data[0]
            if len(data_shape) < 4:
                if len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape))):
                    return np.resize(data, layer_shape)
            return np.transpose(data, layout) if layout is not None else data
        if len(layer_shape) == 2:
            if len(data_shape) == 1:
                return np.transpose([data])
            if len(data_shape) > 2:
                if all(dim == 1 for dim in layer_shape) and all(dim == 1 for dim in data_shape):
                    return np.resize(data, layer_shape)
                if len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape))):
                    return np.resize(data, layer_shape)
        if len(layer_shape) == 3 and len(data_shape) == 4:
            return np.transpose(data, layout)[0] if layout is not None else data[0]
        if layout is not None and len(layer_shape) == len(layout):
            return np.transpose(data, layout)
        if (
                len(layer_shape) == 1 and len(data_shape) > 1 and
                len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape)))
        ):
            return np.resize(data, layer_shape)
        return np.array(data)

    def _align_data_shape(self, data, input_blob):
        input_shape = self.inputs[input_blob]
        data_batch_size = data.shape[0]
        input_batch_size = input_shape[0]
        if data_batch_size < input_batch_size:
            warning_message = 'data batch {} is not equal model input batch_size {}.'.format(
                data_batch_size, input_batch_size
            )
            warning(warning_message)
            diff_number = input_batch_size - data_batch_size
            filled_part = [data[-1]] * diff_number
            data = np.concatenate([data, filled_part])
        return data.reshape(input_shape)

    def automatic_model_search(self):
        def get_xml(model_dir):
            models_list = list(model_dir.glob('{}.xml'.format(self._model_name)))
            if not models_list:
                models_list = list(model_dir.glob('*.xml'))
            return models_list

        def get_blob(model_dir):
            blobs_list = list(Path(model_dir).glob('{}.blob'.format(self._model_name)))
            if not blobs_list:
                blobs_list = list(Path(model_dir).glob('*.blob'))
            return blobs_list

        def get_model():
            model = Path(self.get_value_from_config('model'))
            model_is_blob = self.get_value_from_config('_model_is_blob')
            if not model.is_dir():
                accepted_suffixes = ['.blob', '.xml']
                if model.suffix not in accepted_suffixes:
                    raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
                print_info('Found model {}'.format(model))
                return model, model.suffix == '.blob'
            if model_is_blob or self.backend == 'mx':
                model_list = get_blob(model)
            else:
                model_list = get_xml(model)
                if not model_list and model_is_blob is None:
                    model_list = get_blob(model)
            if not model_list:
                raise ConfigError('suitable model is not found')
            if len(model_list) != 1:
                raise ConfigError('More than one model matched, please specify explicitly')
            model = model_list[0]
            print_info('Found model {}'.format(model))
            return model, model.suffix == '.blob'

        model, is_blob = get_model()
        if is_blob:
            return model, None
        weights = self.get_value_from_config('weights')
        if weights is None or Path(weights).is_dir():
            weights_dir = weights or model.parent
            weights = Path(weights_dir) / model.name.replace('xml', 'bin')
        if weights is not None:
            accepted_weights_suffixes = ['.bin']
            if weights.suffix not in accepted_weights_suffixes:
                raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
            print_info('Found weights {}'.format(get_path(weights)))

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
        for input_blobs in inputs:
            input_data = [input_blobs[input_name] for input_name in self.inputs if input_name not in self.const_inputs]
            outputs = self.comp.apply(cv2.gin(*input_data), args=self.network_args)
            dict_result = dict(zip(self.output_names, outputs))
            results.append(dict_result)

        if metadata is not None:
            for meta_ in metadata:
                meta_['input_shape'] = self.inputs_info_for_meta()

        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError('G-API Launcher does not support async mode yet')

    @staticmethod
    def get_inputs_from_config(config):
        inputs = config.get('inputs')
        if not inputs:
            raise ConfigError('inputs should be provided in config')

        def parse_shape_value(shape):
            return (1, *map(int, get_or_parse_value(shape, ())))
        input_shapes, const_inputs = {}, {}
        for input_param in inputs:
            if input_param['type'] == 'CONST_INPUT':
                const_val = np.array(input_param['value'])
                if const_val.dtype in [float, np.float]:
                    const_val = const_val.astype(np.float32)
                const_inputs[input_param['name']] = const_val
            input_shapes[input_param['name']] = parse_shape_value(input_param.get('shape', []))

        return input_shapes, const_inputs

    def release(self):
        """
        Releases launcher.
        """
        del self.network_args
        del self.comp
