"""
Copyright (c) 2018 Intel Corporation

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

from pathlib import Path
import os
import numpy as np

from ..config import ConfigError, PathField, StringField, DictField, ListField
from ..logging import warning
from ..utils import contains_all, parse_inputs, check_user_inputs, reshape_user_inputs, extract_image_representations
from .launcher import Launcher, LauncherConfig
from .model_conversion import convert_model
from ..logging import print_info
try:
    import openvino.inference_engine as ie
except ImportError:
    import inference_engine as ie

INFERENCE_ENGINE_PLUGINS = [plugin for plugin in ie.known_plugins]
MODEL_OPTIMIZER_DIRECTORY_ENV = 'MO_DIR'

DEVICE_REGEX = r"(?:^HETERO:(?P<devices>(?:{devices})(?:,(?:{devices}))*)$)|(?:^(?P<device>{devices})$)".format(
    devices="|".join(plugin for plugin in ie.known_plugins))


class DLSDKLauncherConfig(LauncherConfig):
    """
    Specifies configuration structure for DLSDK launcher
    """
    model = PathField(check_exists=True, is_directory=False, optional=True)
    weights = PathField(check_exists=True, is_directory=False, optional=True)
    caffe_model = PathField(check_exists=True, is_directory=False, optional=True)
    caffe_weights = PathField(check_exists=True, is_directory=False, optional=True)
    mxnet_weights = PathField(check_exists=True, is_directory=False, optional=True)
    tf_model = PathField(check_exists=True, is_directory=False, optional=True)
    onnx_model = PathField(check_exists=True, is_directory=False, optional=True)
    kaldi_model = PathField(check_exists=True, is_directory=False, optional=True)
    device = StringField(regex=DEVICE_REGEX)
    cpu_extensions = PathField(optional=True)
    gpu_extensions = PathField(optional=True)
    bitstream = PathField(optional=True)
    mo_params = DictField(optional=True)
    mo_flags = ListField(optional=True)
    converted_model_dir = PathField(optional=True)
    outputs = ListField(optional=True)

    _converted_models = PathField(optional=True)
    _models_prefix = PathField(optional=True)
    _model_optimizer = PathField(optional=True, allow_none=True)
    _tf_custom_config_dir = PathField(optional=True, allow_none=True)

    def __init__(self, config_uri, **kwargs):
        super().__init__(config_uri, **kwargs)
        self.need_conversion = None

    def validate(self, entry, field_uri=None):
        """
        Args:
            entry: launcher configuration file entry
            field_uri: id of launcher entry
        Validate that launcher entry meets all configuration structure requirements
        """
        dlsdk_model_options = ['model', 'weights']
        caffe_model_options = ['caffe_model', 'caffe_weights']
        mxnet_model_options = ['mxnet_weights']
        tf_model_options = ['tf_model']
        onnx_model_options = ['onnx_model']
        kaldi_model_options = ['kaldi_model']

        multiple_model_sources_err = ('Either model and weights or caffe_model and caffe_weights ' +
                                      'or mxnet_weights or tf_model should be specified.')
        sources = {'dlsdk': dlsdk_model_options, 'caffe': caffe_model_options, 'tf': tf_model_options,
                   'mxnet': mxnet_model_options, 'onnx': onnx_model_options, 'kaldi': kaldi_model_options}
        specified = []
        for mo_source_option in sources:
            if contains_all(entry, sources[mo_source_option]):
                specified.append(mo_source_option)
        if not specified:
            raise ConfigError('{} None provided'.format(multiple_model_sources_err))
        if len(specified) > 1:
            raise ConfigError('{} Several provided'.format(multiple_model_sources_err))
        self._set_model_source(specified[0])
        super().validate(entry, field_uri)

    def _set_model_source(self, framework):
        self.need_conversion = framework != 'dlsdk'
        self.framework = framework
        self._fields['model'].optional = self.need_conversion
        self._fields['weights'].optional = self.need_conversion
        self._fields['caffe_model'].optional = framework != 'caffe'
        self._fields['caffe_weights'].optional = framework != 'caffe'
        self._fields['mxnet_weights'].optional = framework != 'mxnet'
        self._fields['tf_model'].optional = framework != 'tf'
        self._fields['onnx_model'].optional = framework != 'onnx'
        self._fields['kaldi_model'].optional = framework != 'kaldi'


class DLSDKLauncher(Launcher):
    """
    Class for infer model using DLSDK framework
    """
    __provider__ = 'dlsdk'

    def __init__(self, config_entry, adapter):
        super().__init__(config_entry, adapter)

        dlsdk_launcher_config = DLSDKLauncherConfig('DLSDK_Launcher')
        dlsdk_launcher_config.validate(self._config)

        self._device = self._config['device'].upper()
        self._prepare_bitstream_firmware(self._config)
        self.plugin = ie.IEPlugin(self._device)
        print_info('Loaded {} plugin version: {}'.format(self.plugin.device, self.plugin.version))

        if dlsdk_launcher_config.need_conversion:
            self._convert_model(self._config, dlsdk_launcher_config.framework)
        else:
            self._model = self._config['model']
            self._weights = self._config['weights']

        if self._config.get('cpu_extensions') and 'CPU' in self._device:
            self.plugin.add_cpu_extension(str(self._config.get('cpu_extensions')))
        if self._config.get('gpu_extensions') and 'GPU' in self._device:
            self.plugin.set_config('CONFIG_FILE', str(self._config.get('gpu_extensions')))
        self.network = ie.IENetwork.from_ir(model=str(self._model), weights=str(self._weights))
        self.original_outputs = self.network.outputs
        outputs = self._config.get('outputs')
        if outputs:
            self.network.add_outputs(outputs)
        self.exec_network = self.plugin.load(network=self.network)
        self._config_inputs = parse_inputs(self._config.get('inputs', []))
        check_user_inputs(self.network.inputs.keys(), self._config_inputs)
        reshape_user_inputs(self._config_inputs, self.exec_network.requests[0].inputs)
        image_inputs = list(filter(lambda input: input not in self._config_inputs, self.inputs.keys()))
        if not image_inputs:
            raise ValueError('image input is not found')
        if len(image_inputs) > 1:
            raise ValueError('topologies with several image inputs are not supported')
        self._image_input_blob = image_inputs[0]
        self._batch = self.network.inputs[self._image_input_blob].shape[0]

    @property
    def inputs(self):
        """
        Returns:
            inputs in NCHW format
        """
        # reverse and omit N
        return {k: v.shape[1:] for k, v in self.network.inputs.items() if k not in self._config_inputs}

    @property
    def batch(self):
        return self._batch

    def predict(self, identifiers, data_representation, *args, **kwargs):
        """
        Args:
            identifiers: list of input data identifiers
            data_representation: list of input data representations, which contain preprocessed data and its metadata
        Returns:
            output of model converted to appropriate representation
        """
        data, metadata = extract_image_representations(data_representation)
        data = np.transpose(data, [0, 3, 1, 2])
        input_shape = self.network.inputs[self._image_input_blob].shape
        if data.shape[0] != self._batch:
            input_shape[0] = data.shape[0]
        res = self.exec_network.infer({self._image_input_blob: data.reshape(input_shape), **self._config_inputs})
        raw_outputs_callback = kwargs.get('output_callback')
        if raw_outputs_callback:
            raw_outputs_callback(res)

        if self.adapter is not None:
            self.adapter.output_blob = self.adapter.output_blob or next(iter(self.original_outputs))
            return self.adapter(res, identifiers, metadata)
        return res

    def _is_fpga(self):
        device = self._device
        if 'HETERO:' in self._device:
            device = self._device[len('HETERO:'):]
        devices = [d.upper() for d in device.split(",")]
        return 'FPGA' in devices

    def _prepare_bitstream_firmware(self, config):
        if not self._is_fpga():
            return

        compiler_mode = os.environ.get('CL_CONTEXT_COMPILER_MODE_INTELFPGA')
        if compiler_mode == '3':
            return

        aocx_variable = 'DLA_AOCX'

        bitstream = config.get('bitstream')
        if bitstream:
            os.environ[aocx_variable] = str(bitstream)

        if not os.environ.get(aocx_variable):
            warning('Warning: {} has not been set'.format(aocx_variable))

    def _convert_model(self, config, framework='caffe'):
        converted_models = config['_converted_models'].resolve()  # type: Path
        if not converted_models.is_dir():
            raise FileNotFoundError("Directory for MO converted models not found.")

        mo_params = config.get('mo_params', {})
        mo_flags = config.get('mo_flags', [])
        output_dir = self._replicate_model_directory(
            (Path(config.get(framework + '_model') if config.get(framework + '_model') is not None
                  else config.get(framework + '_weights'))),
            Path(config['_models_prefix']),
            converted_models,
            config.get('converted_model_dir'),
            mo_params.get('output_dir')
        )

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        mo_search_paths = []
        if config.get('_model_optimizer'):
            mo_search_paths.append(config['_model_optimizer'])
        if os.environ.get(MODEL_OPTIMIZER_DIRECTORY_ENV):
            mo_search_paths.append(os.environ[MODEL_OPTIMIZER_DIRECTORY_ENV])
        model_for_conversion = config.get(framework + '_model')
        model_for_conversion = Path(model_for_conversion) if model_for_conversion is not None else None
        weights_for_conversion = config.get(framework + '_weights')
        weights_for_conversion = Path(weights_for_conversion) if weights_for_conversion is not None else None
        topology_name = (model_for_conversion.name.split('.')[0] if model_for_conversion is not None
                         else weights_for_conversion.name.split('.')[0])
        tf_config_dir = config.get('_tf_custom_config_dir')
        self._model, self._weights = convert_model(
            topology_name, output_dir, model_for_conversion,
            weights_for_conversion, framework,
            mo_search_paths, mo_params, mo_flags, tf_config_dir
        )

    @staticmethod
    def _replicate_model_directory(source_path: Path, prefix: Path, destination_path: Path,
                                   postfix=None, mo_output=None) -> Path:
        if postfix is not None and mo_output is not None:
            raise ConfigError(
                'Provided several paths for storage output model. '
                'Please, use only one from these approaches: '
                'specify converted_model_dir parameter or set output_dir in mo_params'
            )
        if mo_output is not None:
            return mo_output
        model_dir = source_path.parent
        suffix = Path(postfix).parts if postfix else model_dir.parts[len(prefix.parts):]
        output_dir = destination_path.joinpath(*suffix)

        return output_dir

    def release(self):
        """
        Releases launcher
        """
        del self.network
        del self.exec_network
        del self.plugin
