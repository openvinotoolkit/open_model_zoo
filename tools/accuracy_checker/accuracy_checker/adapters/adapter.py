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

from ..config import BaseField, ConfigValidator, StringField, ConfigError
from ..dependency import ClassProvider, UnregisteredProviderException
from ..utils import get_parameter_value_from_config


class Adapter(ClassProvider):
    """
    Interface that describes converting raw output to appropriate representation.
    """

    __provider_type__ = 'adapter'

    def __init__(self, launcher_config, label_map=None, output_blob=None):
        self.launcher_config = launcher_config
        self.output_blob = output_blob
        self.label_map = label_map

        self.validate_config(launcher_config)
        self.configure()

    def get_value_from_config(self, key):
        return get_parameter_value_from_config(self.launcher_config, self.parameters(), key)

    @classmethod
    def parameters(cls):
        return {
            'type': StringField(
                default=cls.__provider__ if hasattr(cls, '__provider__') else None, description='Adapter type.'
            ),
        }

    def process(self, raw, identifiers, frame_meta):
        raise NotImplementedError

    def configure(self):
        pass

    @classmethod
    def validate_config(cls, config, fetch_only=False, uri_prefix='', **kwargs):
        if cls.__name__ == Adapter.__name__:
            errors = []
            adapter_type = config if isinstance(config, str) else config.get('type')
            if not adapter_type:
                error = ConfigError(
                    'type is not provided', config, uri_prefix or 'adapter', validation_scheme=cls.validation_scheme()
                )
                if not fetch_only:
                    raise error
                errors.append(error)
                return errors
            try:
                adapter_cls = cls.resolve(adapter_type)
                adapter_config = config if isinstance(config, dict) else {'type': adapter_type}
                return adapter_cls.validate_config(adapter_config, fetch_only=fetch_only, uri_prefix=uri_prefix)
            except UnregisteredProviderException as exception:
                if not fetch_only:
                    raise exception
                return errors
        if 'on_extra_argument' not in kwargs:
            kwargs['on_extra_argument'] = ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
        uri = '{}.{}'.format(uri_prefix, cls.__provider__) if uri_prefix else 'adapter.{}'.format(cls.__provider__)
        return ConfigValidator(uri, fields=cls.parameters(), **kwargs).validate(
            config, fetch_only=fetch_only, validation_scheme=cls.validation_scheme()
        )

    @staticmethod
    def _extract_predictions(outputs_list, meta):
        if isinstance(outputs_list, dict):
            return outputs_list
        return outputs_list[0]

    def select_output_blob(self, outputs):
        if self.output_blob is None:
            self.output_blob = next(iter(outputs))

    @classmethod
    def validation_scheme(cls, provider=None):
        if cls.__name__ == Adapter.__name__:
            if provider:
                return cls.resolve(provider).validation_scheme()
            full_scheme = {}
            for provider_ in cls.providers:
                full_scheme[provider_] = cls.resolve(provider_).validation_scheme()
            return full_scheme
        return cls.parameters()

    def reset(self):
        pass

    def release(self):
        pass

class AdapterField(BaseField):
    def validate(self, entry, field_uri_=None, fetch_only=False, validation_scheme=None):
        errors_stack = super().validate(entry, field_uri_, fetch_only, validation_scheme)

        if entry is None:
            return errors_stack

        field_uri_ = field_uri_ or self.field_uri
        if isinstance(entry, str):
            errors_stack.extend(
                StringField(choices=Adapter.providers).validate(
                    entry, field_uri_ or 'adapter', fetch_only=fetch_only, validation_scheme=validation_scheme
                )
            )
        elif isinstance(entry, dict):
            class DictAdapterValidator(ConfigValidator):
                type = StringField(choices=Adapter.providers)

            dict_adapter_validator = DictAdapterValidator(
                field_uri_ or 'adapter', on_extra_argument=DictAdapterValidator.IGNORE_ON_EXTRA_ARGUMENT
            )
            errors_stack.extend(dict_adapter_validator.validate(
                entry, field_uri_ or 'adapter', fetch_only=fetch_only, validation_scheme=validation_scheme
            ))
        else:
            if not fetch_only:
                errors_stack.append(
                    self.build_error(
                        entry, field_uri_ or 'adapter', 'adapter must be either string or dictionary', validation_scheme
                    ))
            else:
                self.raise_error(entry, field_uri_ or 'adapter', 'adapter must be either string or dictionary')
        return errors_stack


REQUIRES_KALDI = ['kaldi_latgen_faster_mapped']


def create_adapter(adapter_config, launcher=None, dataset=None, delayed_model_loading=False):
    label_map = None
    if dataset:
        metadata = dataset.metadata
        if metadata:
            label_map = metadata.get('label_map')

    if not isinstance(adapter_config, (str, dict)):
        raise ConfigError('Unknown type for adapter configuration')

    adapter_type = adapter_config if isinstance(adapter_config, str) else adapter_config['type']
    adapter_config = adapter_config if isinstance(adapter_config, dict) else {}
    if adapter_type in REQUIRES_KALDI and launcher:
        kaldi_bin_dir = launcher.config.get('_kaldi_bin_dir')
        kaldi_log_file = launcher.config.get('_kaldi_log_file')
        if kaldi_bin_dir:
            adapter_config['_kaldi_bin_dir'] = kaldi_bin_dir
        if kaldi_log_file:
            adapter_config['_kaldi_log_file'] = kaldi_log_file

    adapter = Adapter.provide(adapter_type, adapter_config, label_map=label_map)

    if launcher and not delayed_model_loading and adapter.output_blob is None:
        adapter.output_blob = launcher.output_blob
    return adapter
