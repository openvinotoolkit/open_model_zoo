"""
Copyright (c) 2018-2022 Intel Corporation

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

from contextlib import contextmanager
import sys
import importlib
from collections import OrderedDict

import numpy as np
from ..config import PathField, StringField, DictField, NumberField, ListField
from .launcher import Launcher

MODULE_REGEX = r'(?:\w+)(?:(?:.\w+)*)'
DEVICE_REGEX = r'(?P<device>cpu$|cuda)?'


class PyTorchLauncher(Launcher):
    __provider__ = 'pytorch'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'module': StringField(regex=MODULE_REGEX, description='Network module for loading'),
            'checkpoint': PathField(
                check_exists=True, is_directory=False, optional=True, description='pre-trained model checkpoint'
            ),
            'python_path': PathField(
                check_exists=True, is_directory=True, optional=True,
                description='appendix for PYTHONPATH for making network module visible in current python environment'
            ),
            'module_args': ListField(optional=True, description='positional arguments for network module'),
            'module_kwargs': DictField(
                key_type=str, validate_values=False, optional=True, default={},
                description='keyword arguments for network module'
            ),
            'device': StringField(default='cpu', regex=DEVICE_REGEX),
            'batch': NumberField(value_type=int, min_value=1, optional=True, description="Batch size.", default=1),
            'output_names': ListField(
                optional=True, value_type=str, description='output tensor names'
            )
        })
        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        try:
            # PyTorch import affects performance of common pipeline
            # it is the reason, why it is imported only when it used
            import torch # pylint: disable=C0415
        except ImportError as import_error:
            raise ValueError("PyTorch isn't installed. Please, install it before using. \n{}".format(
                import_error.msg)) from import_error
        self._torch = torch
        self.validate_config(config_entry)
        module_args = config_entry.get("module_args", ())
        module_kwargs = config_entry.get("module_kwargs", {})
        self.device = self.get_value_from_config('device')
        self.cuda = 'cuda' in self.device
        self.module = self.load_module(
            config_entry['module'],
            module_args,
            module_kwargs,
            config_entry.get('checkpoint'),
            config_entry.get('state_key'),
            config_entry.get("python_path")
        )

        self._batch = self.get_value_from_config('batch')
        # torch modules does not have input information
        self._generate_inputs()
        self.output_names = self.get_value_from_config('output_names') or ['output']

    def _generate_inputs(self):
        config_inputs = self.config.get('inputs')
        if not config_inputs:
            self._inputs = {'input': (self.batch, ) + (-1, ) * 3}
            return
        input_shapes = OrderedDict()
        for input_description in config_inputs:
            input_shapes[input_description['name']] = input_description.get('shape', (self.batch, ) + (-1, ) * 3)
        self._inputs = input_shapes

    @property
    def inputs(self):
        return self._inputs

    @property
    def batch(self):
        return self._batch

    @property
    def output_blob(self):
        return next(iter(self.output_names))

    def load_module(self, model_cls, module_args, module_kwargs, checkpoint=None, state_key=None, python_path=None):
        module_parts = model_cls.split(".")
        model_cls = module_parts[-1]
        model_path = ".".join(module_parts[:-1])
        with append_to_path(python_path):
            model_cls = importlib.import_module(model_path).__getattribute__(model_cls)
            module = model_cls(*module_args, **module_kwargs)
            if checkpoint:
                checkpoint = self._torch.load(
                    checkpoint, map_location=None if self.cuda else self._torch.device('cpu')
                )
                state = checkpoint if not state_key else checkpoint[state_key]
                if all(key.startswith('module.') for key in state):
                    module = self._torch.nn.DataParallel(module)
                module.load_state_dict(state, strict=False)
            module.to(self.device)
            module.eval()
            return module

    def fit_to_input(self, data, layer_name, layout, precision, template=None):
        if layout is not None:
            data = np.transpose(data, layout)
        tensor = self._torch.from_numpy(data.astype(np.float32 if not precision else precision))
        tensor = tensor.to(self.device)
        return tensor

    def predict(self, inputs, metadata=None, **kwargs):
        results = []
        with self._torch.no_grad():
            for batch_input in inputs:
                outputs = list(self.module(*batch_input.values()))
                result_dict = {
                    output_name: res.data.cpu().numpy() if self.cuda else res.data.numpy()
                    for output_name, res in zip(self.output_names, outputs)
                }
                results.append(result_dict)
                for meta_ in metadata:
                    meta_['input_shape'] = {key: list(data.shape) for key, data in batch_input.items()}

        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError('PyTorch Launcher does not support async mode yet')

    def release(self):
        del self.module


@contextmanager
def append_to_path(path):
    if path:
        sys.path.append(str(path))

    yield

    if path:
        sys.path.remove(str(path))
