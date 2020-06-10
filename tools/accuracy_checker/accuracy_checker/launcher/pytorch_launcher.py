from contextlib import contextmanager
import sys
import importlib
from collections import OrderedDict

import numpy as np
from ..config import PathField, StringField, DictField, NumberField, ListField
from .launcher import Launcher, LauncherConfigValidator

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
            'module_kwargs':  DictField(
                key_type=str, validate_values=False, optional=True, default={},
                description='keyword arguments for network module'
            ),
            'device': StringField(default='cpu', regex=DEVICE_REGEX),
            'batch': NumberField(value_type=float, min_value=1, optional=True, description="Batch size.", default=1),
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
            raise ValueError("PyTorch isn't installed. Please, install it before using. \n{}".format(import_error.msg))
        self._torch = torch
        pytorch_launcher_config = LauncherConfigValidator('Pytorch_Launcher', fields=self.parameters())
        pytorch_launcher_config.validate(self.config)
        module_args = config_entry.get("module_args", ())
        module_kwargs = config_entry.get("module_kwargs", {})
        self.cuda = 'cuda' in self.get_value_from_config('device')
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
                checkpoint = self._torch.load(checkpoint,
                                              map_location=None if self.cuda else self._torch.device('cpu'))
                state = checkpoint if not state_key else checkpoint[state_key]
                module.load_state_dict(state, strict=False)
            if self.cuda:
                module.cuda()
            else:
                module.cpu()
            module.eval()
            return module

    def fit_to_input(self, data, layer_name, layout, precision):
        data = np.transpose(data, layout)
        tensor = self._torch.from_numpy(data.astype(np.float32 if not precision else precision))
        if self.cuda:
            tensor = tensor.cuda()
        with self._torch.no_grad():
            return self._torch.autograd.Variable(tensor)

    def predict(self, inputs, metadata=None, **kwargs):
        results = []
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
