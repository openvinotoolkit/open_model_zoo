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

from pathlib import Path
from collections import OrderedDict

import cv2

from ..config import PathField, StringField, ConfigError, ListInputsField, ListField
from ..logging import print_info
from .launcher import Launcher, LauncherConfigValidator
from ..utils import get_or_parse_value, get_path


class OpenCVLauncherConfigValidator(LauncherConfigValidator):
    def validate(self, entry, field_uri=None, fetch_only=False):
        self.fields['inputs'].optional = self.delayed_model_loading
        error_stack = super().validate(entry, field_uri)
        if not self.delayed_model_loading:
            inputs = entry.get('inputs')
            for input_layer in inputs:
                if 'shape' not in input_layer:
                    if not fetch_only:
                        raise ConfigError('input value should have shape field')
                    error_stack.extend(self.build_error(entry, field_uri, 'input value should have shape field'))
        return error_stack


class GAPILauncher(Launcher):
    __provider__ = 'g-api'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model file."),
            'weights': PathField(description="Path to weights file.", optional=True, default='', check_exists=False),
            'device': StringField(
                choices=['CPU', 'MYRIAD'],
                description="Device name"
            ),
            'inputs': ListInputsField(optional=False, description="Inputs."),
            'outputs': ListField(value_type=str, allow_empty=False, description='Outputs.')
        })

        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)
        self.validate_config(config_entry, delayed_model_loading=self._delayed_model_loading)
        self.device = self.get_value_from_config('device').upper()
        self.comp = None
        self.network_args = None
        self.output_names = self.get_value_from_config('outputs')
        self._inputs_shapes = self.get_inputs_from_config(self.config)

        if not self._delayed_model_loading:
            self.model, self.weights = self.automatic_model_search()
            self.prepare_net()

    @classmethod
    def validate_config(cls, config, fetch_only=False, delayed_model_loading=False, uri_prefix=''):
        return OpenCVLauncherConfigValidator(
            uri_prefix or 'launcher.{}'.format(cls.__provider__),
            fields=cls.parameters(), delayed_model_loading=delayed_model_loading
        ).validate(config, fetch_only=fetch_only)

    def prepare_net(self):
        inputs = cv2.GInferInputs()
        g_inputs = []
        for input_name in self.inputs:
            g_in = cv2.GMat()
            inputs.setInput(input_name, g_in)
            g_inputs.append(g_in)

        outputs = cv2.gapi.infer("net", inputs)
        g_outputs = [outputs.at(out_name) for out_name in self.output_names]
        self.comp = cv2.GComputation(cv2.GIn(*g_inputs), cv2.GOut(*g_outputs))
        pp = cv2.gapi.ie.params("net", str(self.model), str(self.weights), self.device.upper())
        self.network_args = cv2.GCompileArg(cv2.gapi.networks(pp))

    @property
    def inputs(self):
        return self._inputs_shapes

    @property
    def batch(self):
        return 1

    @property
    def output_blob(self):
        return next(iter(self.output_names))

    def fit_to_input(self, data, layer_name, layout, precision):
        if len(self.inputs) == 1 and self.batch == 1:
            return data[0]
        raise ConfigError('this case is not supported')

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

        def get_onnx(model_dir):
            onnx_list = list(Path(model_dir).glob('{}.onnx'.format(self._model_name)))
            if not onnx_list:
                onnx_list = list(Path(model_dir).glob('*.onnx'))
            return onnx_list

        def get_model():
            model = Path(self.get_value_from_config('model'))
            model_is_blob = self.get_value_from_config('_model_is_blob')
            if not model.is_dir():
                accepted_suffixes = ['.blob', '.onnx', '.xml']
                if model.suffix not in accepted_suffixes:
                    raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
                print_info('Found model {}'.format(model))
                return model, model.suffix == '.blob'
            if model_is_blob:
                model_list = get_blob(model)
            else:
                model_list = get_xml(model)
                if not model_list and model_is_blob is None:
                    model_list = get_blob(model)
                if not model_list:
                    model_list = get_onnx(model)
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
        if (weights is None or Path(weights).is_dir()) and model.suffix != '.onnx':
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
            input_data = [input_blobs[input_name] for input_name in self.inputs]
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

        return OrderedDict([(elem.get('name'), parse_shape_value(elem.get('shape'))) for elem in inputs])

    def release(self):
        """
        Releases launcher.
        """
        del self.network_args
        del self.comp
