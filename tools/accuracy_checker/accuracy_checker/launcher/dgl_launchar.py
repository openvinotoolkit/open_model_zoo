import os
import sys
from collections import OrderedDict

from ..config import NumberField, StringField, BoolField
from ..config import PathField, StringField, NumberField, BoolField, ConfigError
from .launcher import Launcher
import importlib.util
from pprint import pprint

import numpy as np

class DGLLauncher(Launcher):
    __provider__ = 'dgl'

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        try:
            import dgl  # pylint: disable=C0415
            self._dgl = dgl
        except ImportError as import_error:
            raise ValueError(
                "DGL isn't installed. Please, install it before using. \n{}".format(
                    import_error.msg
                )
            )

        try:
            import torch  # pylint: disable=C0415
            self._torch = torch
        except ImportError as import_error:
            raise ValueError(
                "Torch isn't installed. Please, install it before using. \n{}".format(
                    import_error.msg
                )
            )

        self.validate_config(config_entry)
        self.device = self._get_device_to_infer(config_entry.get('device'))  # конфиг это параметры launchers из accuracy-check.yml

        self.module = self.load_module(
            config_entry.get('model'),
            config_entry.get('module'),
            config_entry.get('module_name')
        )

    def _get_device_to_infer(self, device):
        if device == 'CPU':
            return self._torch.device('cpu')
        elif device == 'GPU':
            return self._torch.device('cuda')
        else:
            raise ValueError('The device is not supported')

    def load_module(self, model_path, module_path, module_name):
        file_type = model_path.split('.')[-1]
        supported_extensions = ['pt']
        if file_type not in supported_extensions:
            raise ValueError(f'The file type {file_type} is not supported')

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        foo = importlib.util.module_from_spec(spec)
        sys.modules[f'{module_name}'] = foo
        spec.loader.exec_module(foo)

        import __main__
        setattr(__main__, module_name, getattr(foo, module_name))
        module = self._torch.load(model_path)
        module.to(self.device)
        module.eval()

        return module
        

    @classmethod
    def parameters(cls):
        """Добавляем доп параметры для запуска
        """
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model.", file_or_directory=True),
            'module': StringField(description='Network module for loading'),
            'device': StringField(default='cpu'),
            'module_name': StringField(description='Network module name')
        })
        return parameters

    def predict(self, inputs, metadata=None, **kwargs):
        features = inputs.ndata['feat']
        with torch.inference_mode():
            predictions = self.module(inputs, features).argmax(dim=1)
        return predictions
    
    def release(self):
        """
        Releases launcher.
        """
        del self.module
