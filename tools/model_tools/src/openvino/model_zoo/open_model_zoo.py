import os
import argparse
import yaml

from pathlib import Path

from  openvino.model_zoo import (
    _configuration, _common, omz_downloader, omz_converter
)
from openvino.model_zoo.download_engine import validation


class Model:
    def __init__(self, name=None, model_path=None, description=None, task_type=None, subdirectory=None, ie=None):
        self.model_path = model_path
        self.description = description
        self.task_type = task_type
        self.name = name
        self.subdirectory = subdirectory

        self.net = None
        self.exec_net = None
        self.ie = ie
        self.device = None
        self._accuracy_config = None
        self._model_config = None

    @classmethod
    def download(cls, model_name, *, precision='FP32', download_dir='models', cache_dir=None, ie=None):
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
                Model download directory. By default creates a folder 'models'
                in current directory and downloads to it.
            cache_dir
                Cache directory.
                The script will place a copy of each downloaded file in the cache, or,
                if it is already there, retrieve it from the cache.
                By default creates a folder '.cache' in current download directory.
            ie
                Inference Engine instance
        '''
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
        return cls(name, model_path, description, task_type, subdirectory, ie)

    @classmethod
    def from_pretrained(cls, model_path, *, task_type=None, ie=None):
        '''
        Loads model from existing .xml, .onnx files.
        Parameters
        ----------
            model_path
                Path to .xml or .onnx file.
            task_type
                Type of task that the model performs.
            ie
                Inference Engine instance
        '''
        model_path = Path(model_path).resolve()

        if not model_path.exists():
            raise ValueError('Path {} to model file does not exist.'.format(model_path))

        if model_path.suffix not in ['.xml', '.onnx']:
            raise ValueError('Unsupported model format {}. Only .xml or .onnx supported.'.format(model_path.suffix))

        description = 'Pretrained model {}'.format(model_path.name)
        name = model_path.name
        model = cls._load_model(cls, name)
        task_type = task_type
        subdirectory = model.subdirectory if model else model_path.parent
        model_path = str(model_path)

        return cls(name, model_path, description, task_type, subdirectory, ie)

    def _load_model(self, model_name):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(all=False, list=None, name=model_name, print_all=False)
        try:
            model = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT,
                        mode=_configuration.ModelLoadingMode.ignore_composite)[0]
        except SystemExit:
            model = None

        return model

    def _load(self, device):
        self.net = self.ie.read_network(self.model_path)
        self.exec_net = self.ie.load_network(self.net, device)
        self.device = device

    def __call__(self, inputs, device='CPU'):
        if self.ie is None:
            raise TypeError('ie is not specified or is of the wrong type. Please check ie is of IECore type.')

        if self.exec_net is None or device != self.device:
            self._load(device)

        input_names = self.net.input_info.keys()
        for input_name in inputs:
            if input_name not in input_names:
                raise ValueError('Unknown input name {}'.format(input_name))

        res = self.exec_net.infer(inputs=inputs)
        return res

    @property
    def accuracy_checker_config(self):
        if self._accuracy_config is None:
            config_path = _common.MODEL_ROOT / self.subdirectory / 'accuracy-check.yml'
            if config_path.exists():
                with config_path.open('rb') as config_file, \
                        validation.deserialization_context('Loading config "{}"'.format(config_path)):
                    self._accuracy_config = yaml.safe_load(config_file)

        return self._accuracy_config

    @property
    def model_config(self):
        if self._model_config is None:
            config_path = _common.MODEL_ROOT / self.subdirectory / 'model.yml'
            if config_path.exists():
                with config_path.open('rb') as config_file, \
                        validation.deserialization_context('Loading config "{}"'.format(config_path)):
                    self._model_config = yaml.safe_load(config_file)

        return self._model_config

    def inputs(self):
        if self.net is None:
            try:
                self.net = self.ie.read_network(self.model_path)
            except AttributeError:
                raise TypeError('ie is not specified or is of the wrong type. Please check ie is of IECore type.')

        input_blobs = self.net.input_info

        return input_blobs

    def outputs(self):
        if self.net is None:
            try:
                self.net = self.ie.read_network(self.model_path)
            except AttributeError:
                raise TypeError('ie is not specified or is of the wrong type. Please check ie is of IECore type.')

        output_blob = self.net.outputs

        return output_blob

    def preferable_input_shape(self, name):
        input_info = self.model_config.get('input_info', [])
        for input in input_info:
            if input['name'] == name:
                return input['shape']

        return None

    def layout(self, name):
        input_info = self.model_config.get('input_info', [])
        for input in input_info:
            if input['name'] == name:
                return input['layout']

        return None

    def input_info(self):
        return self.model_config.get('input_info', [])
