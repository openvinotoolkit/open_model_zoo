# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import yaml

from pathlib import Path

from omz_tools import (
    _configuration, _common, omz_downloader, omz_converter
)
from omz_tools.download_engine import validation


def list_models():
    return {model.name for model in _configuration.load_models(_common.MODEL_ROOT, _configuration.ModelLoadingMode.ignore_composite)}


class OMZModel:
    def __init__(
        self, name=None, model_path=None, description=None, task_type=None,
        subdirectory=None, download_dir=None
    ):
        self.model_path = model_path
        self.description = description
        self.task_type = task_type
        self.name = name
        self.subdirectory = subdirectory
        self.download_dir = download_dir

        self._set_config_paths()

    def _set_config_paths(self):
        accuracy_config_path = _common.MODEL_ROOT / self.subdirectory / 'accuracy-check.yml'
        self.accuracy_config_path = accuracy_config_path if accuracy_config_path.exists() else None

        model_config_path = _common.MODEL_ROOT / self.subdirectory / 'model.yml'
        self.model_config_path = model_config_path if model_config_path.exists() else None

    @classmethod
    def download(cls, model_name, *, precision='FP16', download_dir=None, cache_dir=None):
        '''
        Downloads target model. If the model has already been downloaded,
        retrieves the model from the cache instead of downloading it again.
        Then creates OpenVINO Intermediate Representation (IR) network model.

        Creates an object which stores information about the downloaded and IR model.

        Parameters
        ----------
            model_name
                Target model name.
            precision
                Target model precisions.
            download_dir
                Model download directory. Uses .cache/omz in home directory by default.
            cache_dir
                Cache directory.
                The script will place a copy of each downloaded file in the cache, or,
                if it is already there, retrieve it from the cache.
                By default creates a folder '.cache' in current download directory.
        '''
        if download_dir is None:
            download_dir = Path.home() / '.cache' / 'omz'
        else:
            download_dir = Path(download_dir)

        cache_dir = cache_dir or download_dir / '.cache'

        model = cls._load_model(cls, model_name)

        model_dir = download_dir / model.subdirectory

        flags = ['--name=' + model_name,
                 '--output_dir=' + str(download_dir),
                 '--cache_dir=' + str(cache_dir),
                 '--precisions=' + str(precision)]

        omz_downloader.download(flags)

        description = '{}\n\n    License: {}'.format(model.description, model.license_url)
        task_type = model.task_type

        if precision not in model.precisions:
            raise ValueError('Incorrect precision value!\n    Current precision value is: \''
                                        + str(precision) + '\'.\n    Allowed precision values for model are: '
                                        + str(model.precisions))

        prefix = Path(model_dir) / precision / model_name

        model_path = str(prefix) + '.xml'
        bin_path = str(prefix) + '.bin'
        name = model.name
        subdirectory = model.subdirectory

        if not os.path.exists(model_path) or not os.path.exists(bin_path):
            omz_converter.converter(['--name=' + model_name, '--precisions=' + precision,
                        '--download_dir=' + str(download_dir)])
        return cls(name, model_path, description, task_type, subdirectory, download_dir)

    def _load_model(self, model_name):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(all=False, list=None, name=model_name, print_all=False)
        model = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT,
                    mode=_configuration.ModelLoadingMode.ignore_composite)[0]

        return model

    def accuracy_checker_config(self):
        accuracy_config = None
        if self.accuracy_config_path is not None:
            with self.accuracy_config_path.open('rb') as config_file, \
                    validation.deserialization_context('Loading config "{}"'.format(self.accuracy_config_path)):
                accuracy_config = yaml.safe_load(config_file)

        return accuracy_config

    def model_config(self):
        model_config = None
        if self.model_config_path is not None:
            with self.model_config_path.open('rb') as config_file, \
                    validation.deserialization_context('Loading config "{}"'.format(self.model_config_path)):
                model_config = yaml.safe_load(config_file)

        return model_config

    @staticmethod
    def load_vocab_file(vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as r:
            if vocab_file.suffix == '.txt':
                return {t.rstrip("\n"): i for i, t in enumerate(r.readlines())}
            elif vocab_file.suffix == '.json':
                return json.load(r)

    def vocab(self):
        model = self._load_model(self.name)
        vocab_dir = self.download_dir / self.subdirectory
        if model:
            for file in model.files:
                if 'vocab' in file.name.name:
                    vocab_path = vocab_dir / file.name
                    break
        else:
            if (vocab_dir / 'vocab.txt').exists():
                vocab_path = vocab_dir / 'vocab.txt'
            else:
                vocab_path = vocab_dir / 'vocab.json'

        if vocab_path.exists():
            return self.load_vocab_file(vocab_path)
        else:
            return None

    def inputs(self):
        if self.model is None:
            try:
                self.model = self.ie.read_model(self.model_path)
            except AttributeError:
                raise TypeError('ie is not specified or is of the wrong type.'
                                'Please check ie is of openvino.Core type.')

        return self.model.inputs

    def preferable_input_shape(self, name):
        input_info = self.model_config().get('input_info', [])
        for input in input_info:
            if input['name'] == name:
                return input['shape']

        return None

    def layout(self, name):
        input_info = self.model_config().get('input_info', [])
        for input in input_info:
            if input['name'] == name:
                return input['layout']

        return None

    def input_info(self):
        return self.model_config().get('input_info', [])
