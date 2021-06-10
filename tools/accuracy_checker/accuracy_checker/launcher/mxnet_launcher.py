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

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .launcher import Launcher, LauncherConfigValidator, ListInputsField
from ..config import PathField, StringField, NumberField, ConfigError
from ..utils import string_to_tuple
from ..logging import print_info

DEVICE_REGEX = r'(?P<device>cpu$|gpu)(_(?P<identifier>\d+))?'


class MxNetLauncherConfigValidator(LauncherConfigValidator):
    def validate(self, entry, field_uri=None, fetch_only=False):
        self.fields['inputs'].optional = self.delayed_model_loading
        error_stack = super().validate(entry, field_uri)
        if not self.delayed_model_loading:
            inputs = entry.get('inputs')
            for input_layer in inputs:
                if 'shape' not in input_layer:
                    if not fetch_only:
                        raise ConfigError('input value should have shape field')
                    error_stack.extend(self.build_error(entry, field_uri, 'input value should have shape field'))
        return error_stack


class MxNetLauncher(Launcher):
    """
    Class for infer model using MXNet framework
    """
    __provider__ = 'mxnet'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(check_exists=True, file_or_directory=True, description="Path to model."),
            'device': StringField(regex=DEVICE_REGEX, description="Device name.", optional=True, default='CPU'),
            'batch': NumberField(value_type=float, min_value=1, optional=True, description="Batch size."),
            'output_name': StringField(optional=True, description="Output name."),
            'inputs': ListInputsField(optional=False, description="Inputs.")
        })
        return parameters

    def __init__(self, config_entry: dict, *args, **kwargs):
        try:
            import mxnet # pylint: disable=C0415
            self.mxnet = mxnet
        except ImportError as import_error:
            raise ValueError(
                "MXNet isn't installed. Please, install it before using. \n{}".format(import_error.msg)
            )
        super().__init__(config_entry, *args, **kwargs)
        self._delayed_model_loading = kwargs.get('delayed_model_loading', False)

        self.validate_config(config_entry, delayed_model_loading=self._delayed_model_loading)
        if not self._delayed_model_loading:
            # Get model name, prefix, epoch
            self.model = self.automatic_model_search()
            model_path, model_file = self.model.parent, self.model.name
            model_name = model_file.rsplit('.', 1)[0]
            model_prefix, model_epoch = model_name.rsplit('-', 1)

            # Get device and set device context
            match = re.match(DEVICE_REGEX, self.config['device'].lower())
            if match.group('device') == 'gpu':
                identifier = match.group('identifier')
                if identifier is None:
                    identifier = 0
                device_context = self.mxnet.gpu(int(identifier))
            else:
                device_context = self.mxnet.cpu()

            # Get batch from config or 1
            self._batch = self.config.get('batch', 1)

            # Get input shapes
            input_shapes = []

            for input_config in self.config['inputs']:
                input_shape = input_config['shape']
                input_shape = string_to_tuple(input_shape, casting_type=int)
                input_shapes.append((input_config['name'], (self._batch, *input_shape)))

            # Load checkpoints
            sym, arg_params, aux_params = mxnet.model.load_checkpoint(
                model_path / model_prefix, int(model_epoch)
            )
            self._inputs = OrderedDict(input_shapes)
            # Create a module
            self.module = mxnet.mod.Module(symbol=sym, context=device_context, label_names=None)
            self.module.bind(for_training=False, data_shapes=input_shapes)
            self.module.set_params(arg_params, aux_params, allow_missing=True)

    @property
    def batch(self):
        return self._batch

    def fit_to_input(self, data, input_layer, layout, precision):
        if layout:
            data = np.transpose(data, layout)
        return self.mxnet.nd.array(data.astype(precision) if precision else data)

    @property
    def inputs(self):
        return self._inputs

    @classmethod
    def validate_config(cls, config, fetch_only=False, delayed_model_loading=False, uri_prefix=''):
        return MxNetLauncherConfigValidator(
            uri_prefix or 'launcher.{}'.format(cls.__provider__), fields=cls.parameters(),
            delayed_model_loading=delayed_model_loading
        ).validate(config, fetch_only=fetch_only)

    def predict(self, inputs, metadata=None, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """
        results = []
        for infer_input in inputs:
            data_iter = self.mxnet.io.NDArrayIter(
                data=infer_input, label=None, batch_size=self.batch)
            data_batch = self.mxnet.io.DataBatch(data=data_iter.data_list)

            # Infer
            self.module.forward(data_batch)
            infer_res = {}
            for layer, out in zip(self.module.output_names, self.module.get_outputs()):
                infer_res[layer.replace('_output', '')] = out.asnumpy()
            results.append(infer_res)

        if metadata is not None:
            for meta_ in metadata:
                meta_['input_shape'] = self.inputs_info_for_meta()

        return results

    def predict_async(self, *args, **kwargs):
        raise ValueError('MXNet Launcher does not support async mode yet')

    @property
    def output_blob(self):
        return self.config.get('output_name', next(iter(self.module.output_names))).replace('_output', '')

    def automatic_model_search(self):
        model = Path(self.get_value_from_config('model'))
        if model.is_dir():
            model_list = list(model.glob('{}*.params'.format(self._model_name)))
            if not model_list:
                model_list = list(model.glob('*.params'))
                if not model_list:
                    raise ConfigError('Suitable model checkpoint not found')

            if len(model_list) != 1:
                raise ConfigError(
                    'Several model checkpoint found, please specify explicitly, which should be used for validation'
                )
            model = model_list[0]
        accepted_suffixes = ['.params']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('Found model {}'.format(model))

        return model

    def release(self):
        """
        Releases launcher
        """
        del self.module
