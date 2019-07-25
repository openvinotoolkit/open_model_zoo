"""
Copyright (c) 2019 Intel Corporation

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

import subprocess
from pathlib import Path
import os
import platform
import numpy as np
from cpuinfo import get_cpu_info
import openvino.inference_engine as ie

from ..config import ConfigError, NumberField, PathField, StringField, DictField, ListField, BoolField
from ..logging import warning
from ..utils import read_yaml, contains_all, get_path, contains_any, get_parameter_value_from_config
from .launcher import Launcher, LauncherConfigValidator
from .model_conversion import convert_model, FrameworkParameters
from ..logging import print_info

HETERO_KEYWORD = 'HETERO:'
FPGA_COMPILER_MODE_VAR = 'CL_CONTEXT_COMPILER_MODE_INTELFPGA'
DEVICE_REGEX = r"(?:^{hetero}(?P<devices>(?:{devices})(?:,(?:{devices}))*)$)|(?:^(?P<device>{devices})$)".format(
    hetero=HETERO_KEYWORD, devices="|".join(plugin for plugin in ie.known_plugins)
)
VPU_PLUGINS = ('HDDL', "MYRIAD")
VPU_LOG_LEVELS = ('LOG_NONE', 'LOG_WARNING', 'LOG_INFO', 'LOG_DEBUG')


class CPUExtensionPathField(PathField):
    def __init__(self, **kwargs):
        super().__init__(is_directory=False, **kwargs)

    def validate(self, entry, field_uri=None):
        if entry is None:
            return

        field_uri = field_uri or self.field_uri
        validation_entry = ''
        try:
            validation_entry = Path(entry)
        except TypeError:
            self.raise_error(entry, field_uri, "values is expected to be path-like")
        is_directory = False
        if validation_entry.parts[-1] == 'AUTO':
            validation_entry = validation_entry.parent
            is_directory = True
        try:
            get_path(validation_entry, is_directory)
        except FileNotFoundError:
            self.raise_error(validation_entry, field_uri, "path does not exist")
        except NotADirectoryError:
            self.raise_error(validation_entry, field_uri, "path is not a directory")
        except IsADirectoryError:
            self.raise_error(validation_entry, field_uri, "path is a directory, regular file expected")


class DLSDKLauncherConfigValidator(LauncherConfigValidator):
    def __init__(self, config_uri, **kwargs):
        super().__init__(config_uri, **kwargs)
        self.need_conversion = None

    def validate(self, entry, field_uri=None):
        """
        Validate that launcher entry meets all configuration structure requirements.

        Args:
            entry: launcher configuration file entry.
            field_uri: id of launcher entry.
        """
        dlsdk_model_options = ['model', 'weights']
        caffe_model_options = ['caffe_model', 'caffe_weights']
        mxnet_model_options = ['mxnet_weights']
        tf_model_options = ['tf_model']
        tf_meta_options = ['tf_meta']
        onnx_model_options = ['onnx_model']
        kaldi_model_options = ['kaldi_model']

        multiple_model_sources_err = (
            'Either model and weights or caffe_model and caffe_weights '
            'or mxnet_weights or tf_model or tf_meta should be specified.'
        )
        sources = {
            FrameworkParameters('dlsdk', False): dlsdk_model_options,
            FrameworkParameters('caffe', False): caffe_model_options,
            FrameworkParameters('tf', False): tf_model_options,
            FrameworkParameters('mxnet', False): mxnet_model_options,
            FrameworkParameters('onnx', False): onnx_model_options,
            FrameworkParameters('kaldi', False): kaldi_model_options,
            FrameworkParameters('tf', True): tf_meta_options
        }

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
        self.need_conversion = framework.name != 'dlsdk'
        self.framework = framework
        self.fields['model'].optional = self.need_conversion
        self.fields['weights'].optional = self.need_conversion
        self.fields['caffe_model'].optional = framework.name != 'caffe'
        self.fields['caffe_weights'].optional = framework.name != 'caffe'
        self.fields['mxnet_weights'].optional = framework.name != 'mxnet'
        self.fields['tf_model'].optional = framework != FrameworkParameters('tf', False)
        self.fields['tf_meta'].optional = framework != FrameworkParameters('tf', True)
        self.fields['onnx_model'].optional = framework.name != 'onnx'
        self.fields['kaldi_model'].optional = framework.name != 'kaldi'


class DLSDKLauncher(Launcher):
    """
    Class for infer model using DLSDK framework.
    """

    __provider__ = 'dlsdk'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model."),
            'weights': PathField(description="Path to model."),
            'device': StringField(regex=DEVICE_REGEX, description="Device name."),
            'caffe_model': PathField(optional=True, description="Path to Caffe model file."),
            'caffe_weights': PathField(optional=True, description="Path to Caffe weights file."),
            'mxnet_weights': PathField(optional=True, description="Path to MxNet weights file."),
            'tf_model': PathField(optional=True, description="Path to TF model file."),
            'tf_meta': PathField(optional=True, description="Path to TF meta file."),
            'onnx_model': PathField(optional=True, description="Path to ONNX model file."),
            'kaldi_model': PathField(optional=True, description="Path to Kaldi model file."),
            'cpu_extensions': CPUExtensionPathField(optional=True, description="Path to CPU extensions."),
            'gpu_extensions': PathField(optional=True, description="Path to GPU extensions."),
            'bitstream': PathField(optional=True, description="Bitream (FPGA only)."),
            'mo_params': DictField(optional=True, description="Model Optimizer parameters."),
            'mo_flags': ListField(optional=True, description="Model Optimizer flags."),
            'outputs': ListField(optional=True, description="Outputs."),
            'allow_reshape_input': BoolField(optional=True, default=False, description="Allows reshape input."),
            'affinity_map': PathField(optional=True, description="Affinity map."),
            'batch': NumberField(value_type=int, min_value=1, optional=True, default=1, description="Batch size."),
            'should_log_cmd': BoolField(optional=True, description="Log Model Optimizer command."),
            'async_mode': BoolField(optional=True, description="Allows asynchronous mode."),
            'num_requests': NumberField(
                value_type=float, optional=True, min_value=1, default=1,
                description="Number of requests (for async mode only)."
            ),
            '_models_prefix': PathField(is_directory=True, optional=True, description="Model prefix."),
            '_model_optimizer': PathField(optional=True, is_directory=True, description="Model optimizer."),
            '_tf_obj_detection_api_config_dir': PathField(
                optional=True, is_directory=True, description="TF Object Detection API Config."
            ),
            '_tf_custom_op_config_dir': PathField(
                optional=True, is_directory=True, description="TF Custom Operation Config."
            ),
            '_tf_obj_detection_api_pipeline_config_path': PathField(
                optional=True, is_directory=False, description="TF Custom Operation Pipeline Config."),
            '_cpu_extensions_mode': StringField(optional=True, description="CPU extensions mode."),
            '_aocl': PathField(optional=True, description="path to aocl (FPGA only)"),
            '_vpu_log_level': StringField(
                optional=True, choices=VPU_LOG_LEVELS, description="VPU LOG level: {}".format(', '.join(VPU_LOG_LEVELS))
            )
        })

        return parameters

    def __init__(self, config_entry):
        super().__init__(config_entry)

        dlsdk_launcher_config = DLSDKLauncherConfigValidator('DLSDK_Launcher', fields=self.parameters())
        dlsdk_launcher_config.validate(self.config)

        self._device = self.config['device'].upper()
        self._set_variable = False
        self._prepare_bitstream_firmware(self.config)

        if dlsdk_launcher_config.need_conversion:
            self._model, self._weights = DLSDKLauncher.convert_model(self.config, dlsdk_launcher_config.framework)
        else:
            self._model = self.get_value_from_config('model')
            self._weights = self.get_value_from_config('weights')

        self._create_ie_plugin()
        self._create_network()
        requests_num = self.get_value_from_config('num_requests')
        self.exec_network = self.plugin.load(network=self.network, num_requests=requests_num)

        self.allow_reshape_input = self.get_value_from_config('allow_reshape_input')
        self._do_reshape = False
        # It is an important switch -- while the FASTER RCNN is not reshaped correctly, the
        # whole network should be recreated during reshape
        self.reload_network = True

    @property
    def inputs(self):
        """
        Returns:
            inputs in NCHW format.
        """
        return self.network.inputs

    @property
    def batch(self):
        return self._batch

    @property
    def output_blob(self):
        return next(iter(self.original_outputs))

    def predict(self, inputs, metadata, *args, **kwargs):
        """
        Args:
            inputs: dictionary where keys are input layers names and values are data for them.
            metadata: metadata of input representations
        Returns:
            raw data from network.
        """
        results = []
        for infer_inputs in inputs:
            if self._do_reshape:
                input_shapes = {layer_name: data.shape for layer_name, data in infer_inputs.items()}
                self._reshape_input(input_shapes)
                for input_name, input_data in infer_inputs.items():
                    infer_inputs[input_name] = self._align_data_shape(input_data, input_name)

            network_inputs_data = {**infer_inputs}

            benchmark = kwargs.get('benchmark')
            if benchmark:
                benchmark(network_inputs_data)

            result = self.exec_network.infer(network_inputs_data)

            raw_outputs_callback = kwargs.get('output_callback')
            if raw_outputs_callback:
                raw_outputs_callback(result)

            results.append(result)
            for meta_ in metadata:
                meta_['input_shape'] = self.inputs_info_for_meta()
        return results

    def predict_async(self, ir, inputs, metadata, *args, **kwargs):
        infer_inputs = inputs[0]
        benchmark = kwargs.get('benchmark')
        if benchmark:
            benchmark(infer_inputs)
        ir.async_infer(inputs=infer_inputs)
        for meta_ in metadata:
            meta_['input_shape'] = self.inputs_info_for_meta()

    def _is_hetero(self):
        return self._device.startswith(HETERO_KEYWORD)

    def _devices_list(self):
        device = self._device
        if HETERO_KEYWORD in self._device:
            device = self._device[len(HETERO_KEYWORD):]

        return [platform_.upper().strip() for platform_ in device.split(',')]

    def _set_affinity(self, affinity_map_path):
        self.plugin.set_initial_affinity(self.network)
        layers = self.network.layers
        for layer, device in read_yaml(affinity_map_path).items():
            if layer not in layers:
                raise ConfigError('Layer \'{layer}\' is not present in network'.format(layer=layer))
            if device not in self._devices_list():
                raise ConfigError(
                    'Device \'{device}\' set for \'{layer}\' layer is not present in '
                    'provided configuration \'{configuration}\''.format(
                        device=device, layer=layer, configuration=self._device
                    )
                )
            layers[layer].affinity = device

    def _is_fpga(self):
        return 'FPGA' in self._devices_list()

    def _is_vpu(self):
        return contains_any(self._devices_list(), VPU_PLUGINS)

    def _prepare_bitstream_firmware(self, config):
        if not self._is_fpga():
            return

        compiler_mode = os.environ.get(FPGA_COMPILER_MODE_VAR)
        if compiler_mode == '3':
            return

        bitstream = config.get('bitstream')
        if bitstream:
            print_info('programming bitstream: {}'.format(bitstream.name))
            aocl_executable = config.get('_aocl')
            if aocl_executable:
                subprocess.run([str(aocl_executable), 'program', 'acl0', str(bitstream)], check=True)
                os.environ[FPGA_COMPILER_MODE_VAR] = '3'
                self._set_variable = True
            else:
                aocx_variable = 'DLA_AOCX'
                previous_bitstream = os.environ.get(aocx_variable)
                if previous_bitstream == str(bitstream):
                    return
                os.environ[aocx_variable] = str(bitstream)
                if not os.environ.get(aocx_variable):
                    warning('Warning: {} has not been set'.format(aocx_variable))

    @staticmethod
    def get_cpu_extension(cpu_extensions, selection_mode):
        def get_cpu_extensions_list(file_format, base_name, selection_mode):
            if not selection_mode:
                default_cpu_extension = file_format.format(base_name)
                extension_list = list(extensions_path.glob(default_cpu_extension))

                if extension_list:
                    return extension_list

                cpu_info_flags = get_cpu_info()['flags']
                supported_flags = ['avx512', 'avx2', 'sse4_1', 'sse4_2']
                cpu_info_flag_to_suffix = {
                    'avx512': 'avx512',
                    'avx2': 'avx2',
                    'sse4_1': 'sse4',
                    'sse4_2': 'sse4'
                }
                for flag in supported_flags:
                    selection_mode = cpu_info_flag_to_suffix[flag]
                    if flag in cpu_info_flags:
                        break

            extension_list = list(extensions_path.glob(file_format.format('{}_{}'.format(base_name, selection_mode))))

            return extension_list

        os_specific_formats = {
            'Darwin': ('lib{}.dylib', 'lib{}.so'),
            'Linux': ('lib{}.so', ),
            'Windows': ('{}.dll', ),
        }

        cpu_extensions_name = cpu_extensions.parts[-1]
        if cpu_extensions_name != 'AUTO':
            return cpu_extensions
        extensions_path = cpu_extensions.parent
        system_name = platform.system()
        file_formats = os_specific_formats.get(system_name)
        if not file_formats:
            raise ConfigError(
                'Accuracy Checker can not automatically find cpu extensions library '
                'for {} platform. Please, set cpu extension library manually.'.format(system_name)
            )

        extension_list = []

        for supported_format in file_formats:
            extension_list = get_cpu_extensions_list(supported_format, 'cpu_extension', selection_mode)
            if extension_list:
                break

        if not extension_list:
            raise ConfigError('suitable CPU extension lib not found in {}'.format(extensions_path))

        return extension_list[0]


    @staticmethod
    def convert_model(config, framework=FrameworkParameters('caffe', False)):
        config_model = config.get('{}_model'.format(framework.name), '')
        config_weights = config.get('{}_weights'.format(framework.name), '')
        config_meta = config.get('{}_meta'.format(framework.name), '')

        mo_search_paths = []
        model_optimizer = get_parameter_value_from_config(config, DLSDKLauncher.parameters(), '_model_optimizer')
        if model_optimizer:
            mo_search_paths.append(model_optimizer)

        model_optimizer_directory_env = os.environ.get('MO_DIR')
        if model_optimizer_directory_env:
            mo_search_paths.append(model_optimizer_directory_env)

        model_name = (
            Path(config_model).name.rsplit('.', 1)[0] or
            Path(config_weights).name.rsplit('.', 1)[0] or
            Path(config_meta).name.rsplit('.', 1)[0]
        )

        should_log_mo_cmd = get_parameter_value_from_config(config, DLSDKLauncher.parameters(), 'should_log_cmd')

        return convert_model(
            model_name,
            config_model, config_weights, config_meta, framework,
            mo_search_paths,
            get_parameter_value_from_config(config, DLSDKLauncher.parameters(), 'mo_params'),
            get_parameter_value_from_config(config, DLSDKLauncher.parameters(), 'mo_flags'),
            get_parameter_value_from_config(config, DLSDKLauncher.parameters(), '_tf_custom_op_config_dir'),
            get_parameter_value_from_config(config, DLSDKLauncher.parameters(),
                                            '_tf_obj_detection_api_pipeline_config_path'),
            should_log_cmd=should_log_mo_cmd
        )

    @property
    def infer_requests(self):
        return self.exec_network.requests

    def _reshape_input(self, shapes):
        if self.reload_network:
            # Should recreate the whole network
            del self.exec_network
            del self.network
            self._create_network(shapes)
        else:
            del self.exec_network
            self.network.reshape(shapes)

        requests_num = self.config.get('num_requests', 1)
        self.exec_network = self.plugin.load(network=self.network, num_requests=requests_num)
        self._do_reshape = False

    def _set_batch_size(self, batch_size):
        # in some cases we can not use explicit property for setting batch size, so we need to use reshape instead
        # save const inputs without changes
        const_inputs_shapes = {
            input_name: self.network.inputs[input_name].shape for input_name in self.const_inputs
        }
        new_non_const_input_shapes = {}
        for layer_name, layer in self.network.inputs.items():
            if layer_name in const_inputs_shapes:
                continue
            layer_shape = layer.shape
            ind_batch = layer.layout.find('N')
            if ind_batch != -1:
                layer_shape[ind_batch] = batch_size
            new_non_const_input_shapes[layer_name] = layer_shape

        self.network.reshape({**const_inputs_shapes, **new_non_const_input_shapes})

    def _align_data_shape(self, data, input_blob):
        input_shape = self.network.inputs[input_blob].shape
        data_batch_size = data.shape[0]
        input_batch_size = input_shape[0]

        if data_batch_size < input_batch_size:
            warning_message = 'data batch {} is not equal model input batch_size {}. '.format(
                data_batch_size, input_batch_size
            )
            warning(warning_message)
            diff_number = input_batch_size - data_batch_size
            filled_part = [data[-1]] * diff_number
            data = np.concatenate([data, filled_part])

        if len(data.shape) > 1 and len(input_shape) > 1 and data.shape[1] != input_shape[1]:
            data = data[:, :input_shape[1]]

        return data.reshape(input_shape)

    def _create_ie_plugin(self, log=True):
        if hasattr(self, 'plugin'):
            del self.plugin
        self.plugin = ie.IEPlugin(self._device)
        if log:
            print_info('IE version: {}'.format(ie.get_version()))
            print_info('Loaded {} plugin version: {}'.format(self.plugin.device, self.plugin.version))

        cpu_extensions = self.config.get('cpu_extensions')
        if cpu_extensions and 'CPU' in self._devices_list():
            selection_mode = self.config.get('_cpu_extensions_mode')
            cpu_extensions = DLSDKLauncher.get_cpu_extension(cpu_extensions, selection_mode)
            self.plugin.add_cpu_extension(str(cpu_extensions))
        gpu_extensions = self.config.get('gpu_extensions')
        if gpu_extensions and 'GPU' in self._devices_list():
            self.plugin.set_config('CONFIG_FILE', str(gpu_extensions))
        if self._is_vpu():
            log_level = self.config.get('_vpu_log_level')
            if log_level:
                self.plugin.set_config({'VPU_LOG_LEVEL': log_level})

    def _create_network(self, input_shapes=None):
        assert self.plugin, "_create_ie_plugin should be called before _create_network"

        self.network = ie.IENetwork(model=str(self._model), weights=str(self._weights))

        self.original_outputs = self.network.outputs
        outputs = self.config.get('outputs')
        if outputs:
            self.network.add_outputs(outputs)

        if input_shapes is not None:
            self.network.reshape(input_shapes)

        self._batch = self.config.get('batch', self.network.batch_size)
        if self._batch != self.network.batch_size:
            self._set_batch_size(self._batch)
        affinity_map_path = self.config.get('affinity_map')
        if affinity_map_path and self._is_hetero():
            self._set_affinity(affinity_map_path)
        elif affinity_map_path:
            warning('affinity_map config is applicable only for HETERO device')

    def inputs_info_for_meta(self):
        return {
            layer_name: layer.shape for layer_name, layer in self.inputs.items()
            if layer_name not in self.const_inputs + self.image_info_inputs
        }

    def fit_to_input(self, data, layer_name, layout):
        def data_to_blob(layer_shape, data):
            data_shape = np.shape(data)
            if len(layer_shape) == 4:
                if len(data_shape) == 5:
                    data = data[0]
                return np.transpose(data, layout)

            if len(layer_shape) == 2 and len(data_shape) == 1:
                return np.transpose([data])

            return np.array(data)

        layer_shape = tuple(self.inputs[layer_name].shape)

        data = data_to_blob(layer_shape, data)

        data_shape = np.shape(data)
        if data_shape != layer_shape:
            if self.allow_reshape_input:
                self._do_reshape = True
                return data

        return self._align_data_shape(data, layer_name)

    def release(self):
        if 'network' in self.__dict__:
            del self.network
        if 'exec_network' in self.__dict__:
            del self.exec_network
        if 'plugin' in self.__dict__:
            del self.plugin
        if self._set_variable:
            del os.environ[FPGA_COMPILER_MODE_VAR]
