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

from ..topology_types import GenericTopology
from ..config import BaseField, ConfigValidator, StringField, ConfigError
from ..dependency import ClassProvider
from ..utils import get_parameter_value_from_config


class Adapter(ClassProvider):
    """
    Interface that describes converting raw output to appropriate representation.
    """

    __provider_type__ = 'adapter'

    topology_types = (GenericTopology, )

    def __init__(self, launcher_config, label_map=None, output_blob=None):
        self.launcher_config = launcher_config
        self.output_blob = output_blob
        self.label_map = label_map

        self.validate_config()
        self.configure()

    def __call__(self, context=None, outputs=None, **kwargs):
        if outputs is not None:
            return self.process(outputs, **kwargs)
        predictions = self.process(context.prediction_batch, context.identifiers_batch, **kwargs)
        context.prediction_batch = predictions
        return context

    def get_value_from_config(self, key):
        return get_parameter_value_from_config(self.launcher_config, self.parameters(), key)

    @classmethod
    def parameters(cls):
        return {
            'type': StringField(
                default=cls.__provider__ if hasattr(cls, '__provider__') else None, description='Adapter type.'
            ),
        }

    def process(self, raw, identifiers=None, frame_meta=None):
        raise NotImplementedError

    def configure(self):
        pass

    def validate_config(self, **kwargs):
        if 'on_extra_argument' not in kwargs:
            kwargs['on_extra_argument'] = ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
        ConfigValidator(self.__class__.__name__, fields=self.parameters(), **kwargs).validate(self.launcher_config)

    @staticmethod
    def _extract_predictions(outputs_list, meta):
        if isinstance(outputs_list, dict):
            return outputs_list
        return outputs_list[0]


class AdapterField(BaseField):
    def validate(self, entry, field_uri_=None):
        super().validate(entry, field_uri_)

        if entry is None:
            return

        field_uri_ = field_uri_ or self.field_uri
        if isinstance(entry, str):
            StringField(choices=Adapter.providers).validate(entry, 'adapter')
        elif isinstance(entry, dict):
            class DictAdapterValidator(ConfigValidator):
                type = StringField(choices=Adapter.providers)
            dict_adapter_validator = DictAdapterValidator(
                'adapter', on_extra_argument=DictAdapterValidator.IGNORE_ON_EXTRA_ARGUMENT
            )
            dict_adapter_validator.validate(entry)
        else:
            self.raise_error(entry, field_uri_, 'adapter must be either string or dictionary')


def create_adapter(adapter_config, launcher=None, dataset=None):
    label_map = None
    if dataset:
        metadata = dataset.metadata
        if metadata:
            label_map = dataset.metadata.get('label_map')
    if isinstance(adapter_config, str):
        adapter = Adapter.provide(adapter_config, {'type': adapter_config}, label_map=label_map)
    elif isinstance(adapter_config, dict):
        adapter = Adapter.provide(adapter_config['type'], adapter_config, label_map=label_map)
    else:
        raise ConfigError('Unknown type for adapter configuration')

    if launcher:
        adapter.output_blob = launcher.output_blob
    return adapter
