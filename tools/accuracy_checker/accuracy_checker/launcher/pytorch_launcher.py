"""
Copyright (c) 2018-2024 Intel Corporation

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
import urllib
import re
from collections import OrderedDict
import numpy as np
from ..config import PathField, StringField, DictField, NumberField, ListField, BoolField
from ..utils import UnsupportedPackage
from .launcher import Launcher
try:
    import transformers
except ImportError as transformers_error:
    transformers = UnsupportedPackage('transformers', transformers_error.msg)

CLASS_REGEX = r'(?:\w+)'
MODULE_REGEX = r'(?:\w+)(?:(?:.\w+)*)'
DEVICE_REGEX = r'(?P<device>cpu$|cuda)?'
CHECKPOINT_URL_REGEX = r'^https?://.*\.pth(\?.*)?(#.*)?$'
SCALAR_INPUTS = ('input_ids', 'input_mask', 'segment_ids', 'attention_mask', 'token_type_ids')

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
            'checkpoint_url': StringField(
                optional=True, regex=CHECKPOINT_URL_REGEX, description='Url link to pre-trained model checkpoint.'
            ),
            'state_key': StringField(optional=True, regex=r'\w+', description='pre-trained model checkpoint state key'),
            'python_path': PathField(
                check_exists=True, is_directory=True, optional=True,
                description='appendix for PYTHONPATH for making network module visible in current python environment'
            ),
            'module_args': ListField(optional=True, description='positional arguments for network module'),
            'module_kwargs': DictField(
                key_type=str, validate_values=False, optional=True, default={},
                description='keyword arguments for network module'
            ),
            'init_method': StringField(
                optional=True, regex=r'\w+', description='Method name to be called for module initialization.'
            ),
            'device': StringField(default='cpu', regex=DEVICE_REGEX),
            'batch': NumberField(value_type=int, min_value=1, optional=True, description="Batch size.", default=1),
            'output_names': ListField(
                optional=True, value_type=str, description='output tensor names'
            ),
            'use_torch_compile': BoolField(
                optional=True, default=False, description='Use torch.compile to optimize the module code'),
            'torch_compile_kwargs': DictField(
                key_type=str, validate_values=False, optional=True, default={},
                description="dictionary of keyword arguments passed to torch.compile"
            ),
            'transformers_class': StringField(
                optional=True, regex=CLASS_REGEX, description='Transformers class name to load pre-trained module.'
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
        self.use_torch_compile = config_entry.get('use_torch_compile', False)
        self.compile_kwargs = config_entry.get('torch_compile_kwargs', {})
        self.tranformers_class = config_entry.get('transformers_class', None)
        backend = self.compile_kwargs.get('backend', None)
        if self.use_torch_compile and backend == 'openvino':
            try:
                importlib.import_module('openvino.torch')  # pylint: disable=C0415, W0611
            except ImportError as import_error:
                raise ValueError("torch.compile is supported from OpenVINO 2023.1\n{}".format(
                    import_error.msg)) from import_error
        module_args = config_entry.get("module_args", ())
        module_kwargs = config_entry.get("module_kwargs", {})
        self.device = self.get_value_from_config('device')
        self.cuda = 'cuda' in self.device

        checkpoint = config_entry.get('checkpoint')
        if checkpoint is None:
            checkpoint = config_entry.get('checkpoint_url')

        python_path = config_entry.get("python_path")

        if self.tranformers_class:
            self.module = self.load_tranformers_module(
                config_entry['module'], python_path
            )
        else:

            self.module = self.load_module(
                config_entry['module'],
                module_args,
                module_kwargs,
                checkpoint,
                config_entry.get('state_key'),
                python_path,
                config_entry.get("init_method")
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

    def load_module(self, model_cls, module_args, module_kwargs, checkpoint=None, state_key=None, python_path=None,
                    init_method=None
    ):
        module_parts = model_cls.split(".")
        model_cls = module_parts[-1]
        model_path = ".".join(module_parts[:-1])
        with append_to_path(python_path):
            model_cls = getattr(importlib.import_module(model_path), model_cls)
            module = model_cls(*module_args, **module_kwargs)
            if init_method is not None:
                if hasattr(model_cls, init_method):
                    init_method = getattr(module, init_method)
                    module = init_method()
                else:
                    raise ValueError(f'Could not call the method {init_method} in the module {model_cls}.')

            if checkpoint:
                if isinstance(checkpoint, str) and re.match(CHECKPOINT_URL_REGEX, checkpoint):
                    checkpoint = urllib.request.urlretrieve(checkpoint)[0]  # nosec B310  # disable urllib-urlopen check
                checkpoint = self._torch.load(
                    checkpoint, map_location=None if self.cuda else self._torch.device('cpu')
                )
                state = checkpoint if not state_key else checkpoint[state_key]
                if all(key.startswith('module.') for key in state):
                    module = self._torch.nn.DataParallel(module)
                module.load_state_dict(state, strict=False)

            return self.prepare_module(module, model_cls)

    def load_tranformers_module(self, pretrained_name, python_path):
        with append_to_path(python_path):
            if isinstance(transformers, UnsupportedPackage):
                transformers.raise_error(self.__class__.__name__)

            model_class = getattr(transformers, self.tranformers_class)
            pretrained_model = python_path if python_path else pretrained_name
            module = model_class.from_pretrained(pretrained_model)

        return self.prepare_module(module, model_class)

    def prepare_module(self, module, model_class):
        module.to('cuda' if self.cuda else 'cpu')
        module.eval()

        if self.use_torch_compile:
            if hasattr(model_class, 'compile'):
                module.compile()
            module = self._torch.compile(module, **self.compile_kwargs)

        return module


    def _convert_to_tensor(self, value, precision):
        if isinstance(value, self._torch.Tensor):
            return value
        if precision is None:
            precision = np.float32

        return self._torch.from_numpy(value.astype(precision)).to(self.device)

    def fit_to_input(self, data, layer_name, layout, precision, template=None):

        if precision is None and layer_name in SCALAR_INPUTS:
            precision = np.int64

        if layer_name == 'input' and isinstance(data[0], dict):
            tensor_dict = {}
            for key, val in data[0].items():
                if isinstance(val, dict):
                    sub_tensor = {}
                    for k, value in val.items():
                        sub_tensor[k] = self._convert_to_tensor(value, precision)
                    tensor_dict[key] = sub_tensor
                else:
                    tensor_dict[key] = self._convert_to_tensor(val, precision)

            return tensor_dict

        data_shape = np.shape(data)

        if layout is not None and len(data_shape) == len(layout):
            data = np.transpose(data, layout)
        else:
            data = np.array(data)

        return self._convert_to_tensor(data, precision)

    def _convert_to_numpy(self, input_dict):
        numpy_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, self._torch.Tensor):
                numpy_dict[key] = value.detach().cpu().numpy()
            else:
                numpy_dict[key] = value
        return numpy_dict


    def forward(self, outputs):
        if hasattr(outputs, 'logits') and 'logits' in self.output_names:
            return {'logits': outputs.logits}
        if hasattr(outputs, 'last_hidden_state') and 'last_hidden_state' in self.output_names:
            return {'last_hidden_state': outputs.last_hidden_state}
        return list(outputs)

    def predict(self, inputs, metadata=None, **kwargs):
        results = []
        with self._torch.no_grad():
            for batch_input in inputs:
                if metadata[0].get('input_is_dict_type') or (isinstance(batch_input, dict) and 'input' in batch_input):
                    outputs = self.module(batch_input['input'])
                else:
                    outputs = self.module(**batch_input)

                    for meta_ in metadata:
                        meta_['input_shape'] = {key: list(data.shape) for key, data in batch_input.items()}

                if metadata[0].get('output_is_dict_type') or isinstance(outputs, dict):
                    result_dict = self._convert_to_numpy(outputs)
                else:
                    result_dict = {
                        output_name: res.data.cpu().numpy() if self.cuda else res.data.numpy()
                        for output_name, res in zip(self.output_names, outputs)
                    }
                results.append(result_dict)

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
