import os
import argparse

from pathlib import Path

from  open_model_zoo.model_tools import (
    _configuration, _common, downloader, converter
)

class Model:
    def __init__(self, model_name, precision, *, download_dir='models', cache_dir=None):
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

        self.xml_path = str(prefix) + '.xml'
        self.bin_path = str(prefix) + '.bin'

        if not os.path.exists(self.xml_path) or not os.path.exists(self.bin_path):
            converter.main(['--name=' + model_name, '--precisions=' + precision,
                        '--download_dir=' + str(download_dir)])
