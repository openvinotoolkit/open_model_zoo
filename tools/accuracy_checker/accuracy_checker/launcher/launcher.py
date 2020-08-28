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

import numpy as np
from ..adapters import AdapterField
from ..config import ConfigValidator, StringField, ListField, ConfigError, InputField, ListInputsField
from ..dependency import ClassProvider
from ..utils import get_parameter_value_from_config


class LauncherConfigValidator(ConfigValidator):
    def __init__(self, config_uri, fields=None, delayed_model_loading=False, **kwarg):
        super().__init__(config_uri, fields=fields, **kwarg)
        self.delayed_model_loading = delayed_model_loading

    def validate(self, entry, field_uri=None):
        if self.delayed_model_loading:
            if 'model' in self.fields:
                self.fields['model'].optional = True
                self.fields['model'].check_exists = False
            if 'weights' in self.fields:
                self.fields['weights'].optional = True
                self.fields['weights'].check_exists = False
        super().validate(entry, field_uri)
        inputs = entry.get('inputs')
        count_non_const_inputs = 0
        if inputs:
            inputs_by_type = {input_type: [] for input_type in InputField.INPUTS_TYPES}
            for input_layer in inputs:
                input_type = input_layer['type']
                inputs_by_type[input_type].append(input_layer['name'])

                if input_type == 'INPUT':
                    input_value = input_layer.get('value')
                    if not input_value and count_non_const_inputs:
                        raise ConfigError('input value should be specified in case of several non constant inputs')
                    count_non_const_inputs += 1

            additional_attributes = {
                '_list_{}s'.format(input_type.lower()): inputs for input_type, inputs in inputs_by_type.items()
            }

            for additional_attribute, values in additional_attributes.items():
                entry[additional_attribute] = values


class Launcher(ClassProvider):
    """
    Interface for inferring model.
    """

    __provider_type__ = 'launcher'

    def __init__(self, config_entry, *args, model_name='', **kwargs):
        self._model_name = model_name
        self.config = config_entry
        self.default_layout = 'NCHW'
        self.const_inputs = self.config.get('_list_const_inputs', [])
        self.image_info_inputs = self.config.get('_list_image_infos', [])
        self._lstm_inputs = self.config.get('_list_lstm_inputs', [])

    @classmethod
    def parameters(cls):
        return {
            'framework': StringField(
                choices=Launcher.providers, default=cls.__provider__ if cls.__provider__ else None,
                description="Framework name."
            ),
            'tags': ListField(allow_empty=False, optional=True, description="Launcher tags."),
            'inputs': ListInputsField(optional=True, description="Inputs."),
            'adapter': AdapterField(optional=True, description="Adapter."),
            '_list_const_inputs': ListField(
                allow_empty=True, optional=True, default=[], description="List of constant inputs."
            ),
            '_list_inputs': ListField(
                allow_empty=True, optional=True, default=[], description="List of inputs."
            ),
            '_list_image_infos': ListField(
                allow_empty=True, optional=True, default=[], description="List of image information inputs."
            ),
            '_list_lstm_inputs': ListField(
                allow_empty=True, optional=True, default=[], description="List of lstm inputs."
            )
        }

    def validate(self):
        LauncherConfigValidator('Launcher', fields=self.parameters()).validate(self.config)

    def get_value_from_config(self, key):
        return get_parameter_value_from_config(self.config, self.parameters(), key)

    def predict(self, inputs, metadata=None, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """

        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    @property
    def batch(self):
        raise NotImplementedError

    @property
    def output_blob(self):
        raise NotImplementedError

    @property
    def inputs(self):
        raise NotImplementedError

    def predict_async(self, *args, **kwargs):
        raise NotImplementedError('Launcher does not support async mode')

    def _provide_inputs_info_to_meta(self, meta):
        meta['input_shape'] = self.inputs

        return meta

    @staticmethod
    def fit_to_input(data, layer_name, layout, precision):
        if len(np.shape(data)) == len(layout):
            data = np.transpose(data, layout)
        else:
            data = np.array(data)
        return data.astype(precision) if precision else data

    def inputs_info_for_meta(self):
        return {
            layer_name: shape for layer_name, shape in self.inputs.items()
            if layer_name not in self.const_inputs + self.image_info_inputs
        }

    @property
    def name(self):
        return self.__provider__


def unsupported_launcher(name, error_message=None):
    class UnsupportedLauncher(Launcher):
        __provider__ = name

        def __init__(self, config_entry, *args, **kwargs):
            super().__init__(config_entry, *args, **kwargs)

            msg = "{launcher} launcher is disabled. Please install {launcher} to enable it.".format(launcher=name)
            raise ValueError(error_message or msg)

        def predict(self, data, meta=None, **kwargs):
            raise NotImplementedError

        def release(self):
            raise NotImplementedError

        @property
        def batch(self):
            raise NotImplementedError

    return UnsupportedLauncher


def create_launcher(launcher_config, model_name='', delayed_model_loading=False, **kwargs):
    """
    Args:
        launcher_config: launcher configuration file entry.
        model_name: evaluation model name
        delayed_model_loading: allows postpone model loading to the launcher
    Returns:
        framework-specific launcher object.
    """

    launcher_config_validator = LauncherConfigValidator(
        'Launcher_validator',
        delayed_model_loading=delayed_model_loading,
        on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT,
    )
    launcher_config_validator.validate(launcher_config)
    config_framework = launcher_config['framework']

    return Launcher.provide(
        config_framework, launcher_config,
        model_name=model_name, delayed_model_loading=delayed_model_loading, **kwargs
    )
