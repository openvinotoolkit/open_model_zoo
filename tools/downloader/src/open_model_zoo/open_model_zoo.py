import os
import argparse

from pathlib import Path

from  open_model_zoo.model_tools import (
    _configuration, _common, downloader, converter
)


class BaseModel:
    def __init__(self):
        self.model_path = None
        self.bin_path = None

        self.net = None
        self.exec_net = None
        self._ie = None
        self.description = None
        self.task_type = None


class Model(BaseModel):
    def __init__(self, model_name, *, precision='FP32', download_dir='models', cache_dir=None):
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
        super().__init__()

        download_dir = Path(download_dir)
        cache_dir = cache_dir or download_dir / '.cache'

        parser = argparse.ArgumentParser()
        args = argparse.Namespace(all=False, list=None, name=model_name, print_all=False)
        topology = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT)[0]

        model_dir = download_dir / topology.subdirectory

        flags = ['--name=' + model_name,
                 '--output_dir=' + str(download_dir),
                 '--cache_dir=' + str(cache_dir),
                 '--precisions=' + str(precision)]

        downloader.main(flags)

        self.description = '{}\n\n    License: {}'.format(topology.description, topology.license_url)
        self.task_type = topology.task_type

        if precision not in topology.precisions:
            raise ValueError('Incorrect precision value!\n    Current precision value is: \''
                                        + str(precision) + '\'.\n    Allowed precision values for model are: '
                                        + str(topology.precisions))

        prefix = Path(model_dir) / precision / model_name

        self.model_path = str(prefix) + '.xml'
        self.bin_path = str(prefix) + '.bin'

        if not os.path.exists(self.model_path) or not os.path.exists(self.bin_path):
            converter.main(['--name=' + model_name, '--precisions=' + precision,
                        '--download_dir=' + str(download_dir)])

    def from_pretrained(self, model_path):
        '''
        Loads model from existing .xml, .bin files.
        Parameters
        ----------
            model_path
                Path to .xml or .onnx file.
        '''
        model_path = Path(model_path).resolve()

        if not model_path.exists():
            raise ValueError('Path {} to model file does not exist.'.format(model_path))

        if model_path.suffix == '.xml':
            self.model_path = model_path
            self.bin_path = model_path.with_suffix('.bin')

            if not self.bin_path.exists():
                raise ValueError('Path {} to .bin file does not exist.'
                    '.bin file should be in the same directory as .xml'.format(self.bin_path)
                )

            self.model_path = str(self.model_path)
            self.bin_path = str(self.bin_path)
        elif model_path.suffix == '.onnx':
            self.model_path = model_path
        else:
            raise ValueError('Unsupported model format {}. Only .xml or .onnx supported.'.format(model_path.suffix))

        self.description = 'Custom model {}'.format(model_path.name)

    def load(self, ie, device='CPU'):
        self._ie = ie
        self.net = ie.read_network(self.model_path)
        self.exec_net = ie.load_network(self.net, device)

    def inference(self, inputs, ie=None, device='CPU'):
        if self.exec_net is None:
            self.load(ie, device)

        res = self.exec_net.infer(inputs=inputs)
        return res
