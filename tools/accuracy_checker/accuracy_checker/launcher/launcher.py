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
from ..adapters import AdapterField, Adapter
from ..config import (
    ConfigValidator, StringField, ListField, ConfigError, InputField, ListInputsField, PathField
)
from ..dependency import ClassProvider, UnregisteredProviderException
from ..utils import get_parameter_value_from_config


class LauncherConfigValidator(ConfigValidator):
    def __init__(self, config_uri, fields=None, delayed_model_loading=False, **kwarg):
        super().__init__(config_uri, fields=fields, **kwarg)
        self.delayed_model_loading = delayed_model_loading

    def validate(self, entry, field_uri=None, fetch_only=False, validation_scheme=None):
        if self.delayed_model_loading:
            if 'model' in self.fields:
                self.fields['model'].optional = True
                self.fields['model'].check_exists = False
            if 'weights' in self.fields:
                self.fields['weights'].optional = True
                self.fields['weights'].check_exists = False
        error_stack = super().validate(
            entry, field_uri or 'launcher', fetch_only=fetch_only, validation_scheme=validation_scheme
        )
        if 'inputs' in entry:
            error_stack.extend(
                self._validate_inputs(
                    entry, fetch_only=fetch_only, field_uri=field_uri or 'launcher',
                    validation_scheme=validation_scheme
                ))

        return error_stack

    def _validate_inputs(self, entry, fetch_only, field_uri='', validation_scheme=None):
        inputs_uri = field_uri + '.inputs' if field_uri else 'inputs'
        inputs = entry.get('inputs')
        error_stack = []
        count_non_const_inputs = 0
        inputs_by_type = {input_type: [] for input_type in InputField.INPUTS_TYPES}
        inputs_valid_scheme = validation_scheme['inputs'] if validation_scheme else validation_scheme
        for input_id, input_layer in enumerate(inputs):
            input_uri = '{}.{}'.format(inputs_uri, input_id)
            input_type = input_layer.get('type')
            if input_type is None:
                reason = 'input type is not provided'
                if not fetch_only:
                    raise ConfigError(reason, input_layer, input_uri)
                error_stack.append(
                    self.build_error(input_layer, input_uri, reason, validation_scheme=inputs_valid_scheme)
                )
                continue
            if input_type not in InputField.INPUTS_TYPES:
                reason = 'undefined input type {}'.format(input_type)
                if not fetch_only:
                    raise ConfigError(reason, input_layer, input_uri)
                error_stack.append(
                    self.build_error(input_layer, input_uri, reason, validation_scheme=inputs_valid_scheme)
                )
                continue
            if 'name' not in input_layer:
                reason = 'input name is not provided'
                if not fetch_only:
                    raise ConfigError(reason, input_layer, input_uri)
                error_stack.append(
                    self.build_error(input_layer, input_uri, reason, validation_scheme=inputs_valid_scheme)
                )
                continue
            inputs_by_type[input_type].append(input_layer['name'])
            if input_type == 'INPUT':
                reason = 'input value should be specified in case of several non constant inputs'
                input_value = input_layer.get('value')
                if not input_value and count_non_const_inputs:
                    if not fetch_only:
                        raise ConfigError(reason)
                    error_stack.append(
                        self.build_error(
                            input_layer,
                            input_uri,
                            reason,
                            validation_scheme=validation_scheme
                        )
                    )
                count_non_const_inputs += 1

        additional_attributes = {
            '_list_{}s'.format(input_type.lower()): inputs for input_type, inputs in inputs_by_type.items()
        }

        for additional_attribute, values in additional_attributes.items():
            entry[additional_attribute] = values

        return error_stack


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
        self._ignore_inputs = self.config.get('_list_ignore_inputs', [])
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)

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
            '_list_orig_image_infos': ListField(
                allow_empty=True, optional=True, default=[], description="List of image information inputs."
            ),
            '_list_lstm_inputs': ListField(
                allow_empty=True, optional=True, default=[], description="List of lstm inputs."
            ),
            '_list_ignore_inputs': ListField(
                allow_empty=True, optional=True, default=[], description='List of ignored inputs'
            ),
            '_input_precision': ListField(
                allow_empty=True, optional=True, default=[], description='Input precision list from command line.'
            ),
            '_kaldi_bin_dir': PathField(is_directory=True, optional=True, description='directory with Kaldi binaries'),
            '_kaldi_log_file': PathField(
                optional=True, description='File for saving Kaldi tools logs', check_exists=False
            )
        }

    @classmethod
    def validation_scheme(cls, provider=None):
        if cls.__name__ == Launcher.__name__:
            if provider:
                return cls.resolve(provider).validation_scheme()
            full_scheme = []
            for provider_ in cls.providers:
                full_scheme.append(cls.resolve(provider_).validation_scheme())
            return full_scheme
        scheme = {}
        for key, value in cls.parameters().items():
            if key.startswith('_'):
                continue
            if key == 'adapter':
                scheme[key] = Adapter
                continue
            scheme[key] = value
        return scheme

    @classmethod
    def validate_config(cls, config, delayed_model_loading=False, fetch_only=False, uri_prefix=''):
        if cls.__name__ == Launcher.__name__:
            errors = []
            framework = config.get('framework')
            if not framework:
                error = ConfigError(
                    'framework is not provided', config, uri_prefix or 'launcher',
                    validation_scheme=cls.validation_scheme()
                )
                if not fetch_only:
                    raise error
                errors.append(error)
                return errors
            try:
                launcher_cls = cls.resolve(framework)
                return launcher_cls.validate_config(config, fetch_only=fetch_only, uri_prefix=uri_prefix)
            except UnregisteredProviderException as exception:
                if not fetch_only:
                    raise exception
                errors.append(
                    ConfigError(
                        "launcher {} is not unregistered".format(framework), config, uri_prefix or 'launcher',
                        validation_scheme=cls.validation_scheme())
                )
                return errors
        uri = uri_prefix or'launcher.{}'.format(cls.__provider__)
        return LauncherConfigValidator(
            uri, fields=cls.parameters(), delayed_model_loading=delayed_model_loading
        ).validate(
            config, fetch_only=fetch_only, field_uri=uri, validation_scheme=cls.validation_scheme()
        )

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
        if layout is not None and len(np.shape(data)) == len(layout):
            data = np.transpose(data, layout)
        else:
            data = np.array(data)
        return data.astype(precision) if precision else data

    def inputs_info_for_meta(self, *args, **kwargs):
        return {
            layer_name: shape for layer_name, shape in self.inputs.items()
            if layer_name not in self.const_inputs + self.image_info_inputs + self._ignore_inputs
        }

    def update_input_configuration(self, input_config):
        self.config['inputs'] = input_config

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
