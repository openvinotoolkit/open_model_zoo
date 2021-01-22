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

from ..utils import get_path, read_txt
from ..logging import print_info
from ..config import PathField, StringField, BoolField
from .loaders import Loader
from .launcher import Launcher


class DummyLauncher(Launcher):
    """
    Class for using predictions from another tool.
    """

    __provider__ = 'dummy'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'loader': StringField(choices=Loader.providers, description="Loader."),
            'data_path': PathField(description="Data path."),
            'provide_identifiers': BoolField(optional=True, default=False),
            'identifiers_list': PathField(optional=True)
        })
        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)

        self.validate_config(config_entry)
        print_info('Predictions objects loading started')
        self.data_path = get_path(self.get_value_from_config('data_path'))
        identfiers_file = self.get_value_from_config('identifiers_list')
        if identfiers_file is not None:
            kwargs['identifiers'] = read_txt(identfiers_file)

        self._loader = Loader.provide(self.get_value_from_config('loader'), self.data_path, **kwargs)

        print_info("\n{} predictions objects loaded from {}".format(len(self._loader), self.data_path))

    def predict(self, identifiers, *args, **kwargs):
        return [self._loader[identifier] for identifier in identifiers]

    def predict_async(self, *args, **kwargs):
        raise ValueError('DummyLauncher does not support async processing')

    def release(self):
        pass

    @property
    def batch(self):
        return 1

    @property
    def inputs(self):
        return {}

    def get_all_inputs(self):
        return self.inputs

    def inputs_info_for_meta(self):
        return {}

    @property
    def output_blob(self):
        return self.data_path
