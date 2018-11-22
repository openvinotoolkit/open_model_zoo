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
from ..config import BaseField
from ..adapters import Adapter, AdapterField
from ..config import ConfigValidator, StringField, ListField
from ..dependency import ClassProvider, provide


class Launcher(ClassProvider):
    """
    Interface for inferring model
    """
    __provider_type__ = 'launcher'

    adapter = provide(Adapter)

    def __init__(self, config_entry, adapter, *args, **kwargs):
        self.adapter = adapter
        self._config = config_entry

    def predict(self, identifiers, data, *args, **kwargs):
        """
        Args:
            identifiers: id of given data
            data: already was preprocessed

        Returns:
            raw data from network
        """
        raise NotImplementedError

    def release(self):
        """
        Releases launcher
        """
        raise NotImplementedError

    @property
    def batch(self):
        """
        Returns:
             batch size
        """
        raise NotImplementedError


class InputValidator(ConfigValidator):
    name = StringField()
    value = BaseField()


class LauncherConfig(ConfigValidator):
    """
    Specifies common part of configuration structure for launchers
    """
    framework = StringField(choices=Launcher.providers)
    inputs = ListField(allow_empty=False, value_type=InputValidator('Inputs'), optional=True)
    adapter = AdapterField()


def unsupported_launcher(name, error_message=None):
    class UnsupportedLauncher(Launcher):
        __provider__ = name

        def __init__(self, config_entry, adapter, *args, **kwargs):
            super().__init__(config_entry, adapter, *args, **kwargs)

            msg = "{launcher} launcher is disabled. Please install {launcher} to enable it.".format(launcher=name)
            raise ValueError(error_message or msg)

        def predict(self, identifiers, data, *args, **kwargs):
            raise NotImplementedError

        def release(self):
            raise NotImplementedError

        @property
        def batch(self):
            raise NotImplementedError

    return UnsupportedLauncher


def create_launcher(launcher_config, dataset_meta=None):
    """
    Args:
        launcher_config: launcher configuration file entry
        dataset_meta: metadata dictionary for dataset annotation
    Returns:
        Framework-specific launcher object
    """

    launcher_config_validator = LauncherConfig('Launcher_validator',
                                               on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)
    launcher_config_validator.validate(launcher_config)

    label_map = None
    if dataset_meta:
        label_map = dataset_meta.get('label_map')

    config_framework = launcher_config['framework']
    config_adapter = launcher_config.get('adapter')
    if not config_adapter:
        adapter = None
    elif isinstance(config_adapter, str):
        adapter = Adapter.provide(config_adapter, launcher_config, label_map=label_map)
    elif isinstance(config_adapter, dict):
        adapter = Adapter.provide(config_adapter['type'], config_adapter, label_map=label_map)
    else:
        raise ValueError

    return Launcher.provide(config_framework, launcher_config, adapter=adapter)
