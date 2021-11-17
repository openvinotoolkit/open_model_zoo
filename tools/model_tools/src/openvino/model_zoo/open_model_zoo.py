import re
import cv2
import os
import argparse
import yaml

from pathlib import Path

from  openvino.model_zoo import (
    _configuration, _common, omz_downloader, omz_converter
)
from openvino.model_zoo.download_engine import validation


class Model:
    def __init__(self, name=None, model_path=None, description=None, task_type=None, subdirectory=None):
        self.model_path = model_path
        self.description = description
        self.task_type = task_type
        self.name = name
        self.subdirectory = subdirectory

        self.net = None
        self.exec_net = None
        self._accuracy_config = None
        self._model_config = None

    @classmethod
    def download_model(cls, model_name, *, precision='FP32', download_dir='models', cache_dir=None):
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
        '''
        download_dir = Path(download_dir)
        cache_dir = cache_dir or download_dir / '.cache'

        parser = argparse.ArgumentParser()
        args = argparse.Namespace(all=False, list=None, name=model_name, print_all=False)
        model = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT)[0]

        model_dir = download_dir / model.subdirectory

        flags = ['--name=' + model_name,
                 '--output_dir=' + str(download_dir),
                 '--cache_dir=' + str(cache_dir),
                 '--precisions=' + str(precision)]

        omz_downloader.main(flags)

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
            omz_converter.main(['--name=' + model_name, '--precisions=' + precision,
                        '--download_dir=' + str(download_dir)])
        return cls(name, model_path, description, task_type, subdirectory)

    @classmethod
    def from_pretrained(cls, model_path, *, task_type=None):
        '''
        Loads model from existing .xml, .onnx files.
        Parameters
        ----------
            model_path
                Path to .xml or .onnx file.
            task_type
                Type of task that the model performs.
        '''
        model_path = Path(model_path).resolve()

        if not model_path.exists():
            raise ValueError('Path {} to model file does not exist.'.format(model_path))

        if model_path.suffix not in ['.xml', '.onnx']:
            raise ValueError('Unsupported model format {}. Only .xml or .onnx supported.'.format(model_path.suffix))

        description = 'Pretrained model {}'.format(model_path.name)
        name = model_path.name
        task_type = task_type
        subdirectory = model_path.parent
        model_path = str(model_path)

        return cls(name, model_path, description, task_type, subdirectory)

    def _load(self, ie, device):
        self.net = ie.read_network(self.model_path)
        self.exec_net = ie.load_network(self.net, device)

    def __call__(self, inputs, ie, device='CPU'):
        if self.exec_net is None:
            self._load(ie, device)

        input_names = next(iter(self.net.input_info))
        for input_name, value in inputs.items():
            if input_name not in input_names:
                raise ValueError('Unknown input name {}'.format(input_name))

            input_shape = self.net.input_info[input_name].input_data.shape
            if input_shape != list(value.shape):
                value = cv2.resize(value, input_shape)
        res = self.exec_net.infer(inputs=inputs)
        return res

    @property
    def accuracy_checker_config(self):
        if self._accuracy_config is None:
            config_path = _common.MODEL_ROOT / self.subdirectory / 'accuracy-check.yml'
            with config_path.open('rb') as config_file, \
                    validation.deserialization_context('Loading config "{}"'.format(config_path)):
                self._accuracy_config = yaml.safe_load(config_file)

        return self._accuracy_config

    @property
    def model_config(self):
        if self._model_config is None:
            config_path = _common.MODEL_ROOT / self.subdirectory / 'model.yml'
            with config_path.open('rb') as config_file, \
                    validation.deserialization_context('Loading config "{}"'.format(config_path)):
                self._model_config = yaml.safe_load(config_file)

        return self._model_config

    def inputs(self, ie=None):
        if self.net is None:
            try:
                self.net = ie.read_network(self.model_path)
            except AttributeError:
                raise TypeError('ie argumnet must be of IECore type.')
        
        input_blob = next(iter(self.net.input_info))

        return input_blob
