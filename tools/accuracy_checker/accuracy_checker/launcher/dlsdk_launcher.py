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
import multiprocessing
from pathlib import Path
import os
import platform
import re
from collections import OrderedDict
import numpy as np
import openvino.inference_engine as ie

from .dlsdk_async_request import AsyncInferRequestWrapper
from ..config import ConfigError, NumberField, PathField, StringField, DictField, ListField, BoolField, BaseField
from ..logging import warning
from ..utils import (
    read_yaml,
    contains_all,
    get_path,
    contains_any,
    get_parameter_value_from_config,
    string_to_tuple,
    get_or_parse_value
)
from .launcher import Launcher, LauncherConfigValidator
from .model_conversion import convert_model, FrameworkParameters
from ..logging import print_info
from .input_feeder import PRECISION_TO_DTYPE, DIM_IDS_TO_LAYOUT
try:
    from cpuinfo import get_cpu_info
except ImportError:
    get_cpu_info = None

try:
    from openvino.inference_engine import Blob, TensorDesc
except ImportError:
    try:
        # old structures names compatibilities
        from openvino.inference_engine import IEBlob, IETensorDesc
        Blob = IEBlob
        TensorDesc = IETensorDesc
    except ImportError:
        Blob, TensorDesc = None, None


HETERO_KEYWORD = 'HETERO:'
MULTI_DEVICE_KEYWORD = 'MULTI:'
FPGA_COMPILER_MODE_VAR = 'CL_CONTEXT_COMPILER_MODE_INTELFPGA'
NIREQ_REGEX = r"(\(\d+\))"
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
    def __init__(
            self, config_uri, fields=None, delayed_model_loading=False, **kwargs
    ):
        super().__init__(config_uri, fields, delayed_model_loading, **kwargs)
        self.need_conversion = None

    def create_device_regex(self, available_devices):
        self.regular_device_regex = r"(?:^(?P<device>{devices})$)".format(devices="|".join(available_devices))
        self.hetero_regex = r"(?:^{hetero}(?P<devices>(?:{devices})(?:,(?:{devices}))*)$)".format(
            hetero=HETERO_KEYWORD, devices="|".join(available_devices)
        )
        self.multi_device_regex = r"(?:^{multi}(?P<devices_ireq>(?:{devices_ireq})(?:,(?:{devices_ireq}))*)$)".format(
            multi=MULTI_DEVICE_KEYWORD, devices_ireq="{}?|".format(NIREQ_REGEX).join(available_devices)
        )
        self.supported_device_regex = r"{multi}|{hetero}|{regular}".format(
            multi=self.multi_device_regex, hetero=self.hetero_regex, regular=self.regular_device_regex
        )
        self.fields['device'].set_regex(self.supported_device_regex)

    def validate(self, entry, field_uri=None, ie_core=None):
        """
        Validate that launcher entry meets all configuration structure requirements.
        Args:
            entry: launcher configuration file entry.
            field_uri: id of launcher entry.
            ie_core: IECore instance.
        """
        if not self.delayed_model_loading:
            framework_parameters = self.check_model_source(entry)
            self._set_model_source(framework_parameters)
        super().validate(entry, field_uri)
        self.create_device_regex(ie.known_plugins)
        try:
            self.fields['device'].validate(entry['device'], field_uri)
        except ConfigError as error:
            if ie_core is not None:
                self.create_device_regex(ie_core.available_devices)
                try:
                    self.fields['device'].validate(entry['device'], field_uri)
                except ConfigError:
                    # workaround for devices where this metric is non implemented
                    warning('unknown device: {}'.format(entry['device']))
            else:
                raise error

    def _set_model_source(self, framework):
        self.need_conversion = framework.name != 'dlsdk'
        self.framework = framework
        self.fields['model'].optional = self.need_conversion
        self.fields['caffe_model'].optional = framework.name != 'caffe'
        self.fields['caffe_weights'].optional = framework.name != 'caffe'
        self.fields['mxnet_weights'].optional = framework.name != 'mxnet'
        self.fields['tf_model'].optional = framework != FrameworkParameters('tf', False)
        self.fields['tf_meta'].optional = framework != FrameworkParameters('tf', True)
        self.fields['onnx_model'].optional = framework.name != 'onnx'
        self.fields['kaldi_model'].optional = framework.name != 'kaldi'

    @staticmethod
    def check_model_source(entry):
        dlsdk_model_options = ['model']
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

        return specified[0]


class DLSDKLauncher(Launcher):
    """
    Class for infer model using DLSDK framework.
    """

    __provider__ = 'dlsdk'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model.", file_or_directory=True),
            'weights': PathField(description="Path to weights.", optional=True, file_or_directory=True),
            'device': StringField(description="Device name."),
            'caffe_model': PathField(optional=True, description="Path to Caffe model file."),
            'caffe_weights': PathField(optional=True, description="Path to Caffe weights file."),
            'mxnet_weights': PathField(optional=True, description="Path to MXNet weights file."),
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
            'async_mode': BoolField(optional=True, description="Allows asynchronous mode.", default=False),
            'num_requests': BaseField(
                optional=True,
                description="Number of requests (for async mode only). "
                            "In multi device mode allows setting comma-separated list for numbers "
                            "or one value which will be used for all devices"
            ),
            '_model_optimizer': PathField(optional=True, is_directory=True, description="Model optimizer."),
            '_tf_obj_detection_api_config_dir': PathField(
                optional=True, is_directory=True, description="TF Object Detection API Config."
            ),
            '_tf_custom_op_config_dir': PathField(
                optional=True, is_directory=True, description="TF Custom Operation Config prefix."
            ),
            '_transformations_config_dir': PathField(
                optional=True, is_directory=True, description="Transformation config prefix for Model Optimizer"),
            '_tf_obj_detection_api_pipeline_config_path': PathField(
                optional=True, is_directory=False, description="TF Custom Operation Pipeline Config."),
            '_cpu_extensions_mode': StringField(optional=True, description="CPU extensions mode."),
            '_aocl': PathField(optional=True, description="path to aocl (FPGA only)"),
            '_vpu_log_level': StringField(
                optional=True, choices=VPU_LOG_LEVELS, description="VPU LOG level: {}".format(', '.join(VPU_LOG_LEVELS))
            ),
            '_prev_bitstream': PathField(optional=True, description="path to bitstream from previous run (FPGA only)"),
            '_device_config': PathField(optional=True, description='path to file with device configuration'),
            '_model_is_blob': BoolField(optional=True, description='hint for auto model search')
        })

        return parameters

    def __init__(self, config_entry, model_name='', delayed_model_loading=False):
        super().__init__(config_entry, model_name)

        self._set_variable = False
        self.ie_config = self.config.get('ie_config')
        self.ie_core = ie.IECore(xml_config_file=str(self.ie_config)) if self.ie_config is not None else ie.IECore()
        self._delayed_model_loading = delayed_model_loading
        dlsdk_launcher_config = DLSDKLauncherConfigValidator(
            'DLSDK_Launcher', fields=self.parameters(), delayed_model_loading=delayed_model_loading,
        )
        dlsdk_launcher_config.validate(self.config, ie_core=self.ie_core)
        device = self.config['device'].split('.')
        self._device = '.'.join((device[0].upper(), device[1])) if len(device) > 1 else device[0].upper()
        self._set_variable = False
        self._async_mode = False
        self._prepare_bitstream_firmware(self.config)
        self._prepare_ie()
        self._delayed_model_loading = delayed_model_loading

        if not delayed_model_loading:
            if dlsdk_launcher_config.need_conversion:
                self._model, self._weights = DLSDKLauncher.convert_model(self.config, dlsdk_launcher_config.framework)
            else:
                self._model, self._weights = self.automatic_model_search()

            self.load_network(log=True)
            self.allow_reshape_input = self.get_value_from_config('allow_reshape_input') and self.network is not None
        else:
            self.allow_reshape_input = self.get_value_from_config('allow_reshape_input')
        self._do_reshape = False
        self._use_set_blob = False
        self._target_layout_mapping = {}

    @property
    def device(self):
        return self._device

    @property
    def inputs(self):
        """
        Returns:
            inputs in NCHW format.
        """
        if self.network is None:
            has_info = hasattr(self.exec_network, 'input_info')
            if not has_info:
                return self.exec_network.inputs
            return OrderedDict([(name, data.input_data) for name, data in self.exec_network.input_info.items()])
        has_info = hasattr(self.network, 'input_info')
        if has_info:
            return OrderedDict([(name, data.input_data) for name, data in self.network.input_info.items()])
        return self.network.inputs

    @property
    def batch(self):
        return self._batch

    @property
    def output_blob(self):
        return next(iter(self.original_outputs))

    def predict(self, inputs, metadata=None, **kwargs):
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
            if self._use_set_blob:
                has_info = hasattr(self.exec_network, 'input_info')
                for key, input_data in infer_inputs.items():
                    if has_info:
                        ie_input_info = OrderedDict([
                            (name, data.input_data) for name, data in self.exec_network.input_info.items()
                        ])
                    else:
                        ie_input_info = self.exec_network.inputs
                    layout = self._target_layout_mapping.get(key, ie_input_info[key].layout)
                    tensor_desc = TensorDesc(
                        ie_input_info[key].precision,
                        ie_input_info[key].shape,
                        layout
                    )
                    self.exec_network.requests[0].set_blob(key, Blob(tensor_desc, input_data))
            result = self.exec_network.infer(infer_inputs) if not self._use_set_blob else self.exec_network.infer()
            results.append(result)

        if metadata is not None:
            for meta_ in metadata:
                meta_['input_shape'] = self.inputs_info_for_meta()
        self._do_reshape = False
        self._use_set_blob = False

        return results

    def predict_async(self, ir, inputs, metadata=None, context=None, **kwargs):
        infer_inputs = inputs[0]
        if metadata is not None:
            for meta_ in metadata:
                meta_['input_shape'] = self.inputs_info_for_meta()
        ir.infer(infer_inputs, metadata, context)

    def _is_hetero(self):
        return self._device.startswith(HETERO_KEYWORD)

    def _is_multi(self):
        return self._device.startswith(MULTI_DEVICE_KEYWORD)

    def _devices_list(self):
        device = self._device
        if self._is_hetero():
            device = self._device[len(HETERO_KEYWORD):]
        if self._is_multi():
            device = self._device[len(MULTI_DEVICE_KEYWORD):]
            device = re.sub(NIREQ_REGEX, '', device)

        return [platform_.upper().strip() for platform_ in device.split(',')]

    def _set_affinity(self, affinity_map_path):
        automatic_affinity = self.ie_core.query_network(self.network, self._device)
        custom_affinity = read_yaml(affinity_map_path)
        for layer in custom_affinity:
            if layer not in automatic_affinity:
                raise ConfigError('Layer \'{layer}\' is not present in network'.format(layer=layer))
        layers = self.network.layers
        for layer_name in layers:
            device = custom_affinity.get(layer_name, automatic_affinity[layer_name])
            if device not in self._devices_list():
                raise ConfigError(
                    'Device \'{device}\' set for \'{layer}\' layer is not present in '
                    'provided configuration \'{configuration}\''.format(
                        device=device, layer=layer_name, configuration=self._device
                    )
                )
            layers[layer_name].affinity = device

    def automatic_model_search(self):
        def get_xml(model_dir):
            models_list = list(model_dir.glob('{}.xml'.format(self._model_name)))
            if not models_list:
                models_list = list(model_dir.glob('*.xml'))
            return models_list

        def get_blob(model_dir):
            blobs_list = list(Path(model_dir).glob('{}.blob'.format(self._model_name)))
            if not blobs_list:
                blobs_list = list(Path(model_dir).glob('*.blob'))
            return blobs_list

        def get_model():
            model = Path(self.get_value_from_config('model'))
            model_is_blob = self.get_value_from_config('_model_is_blob')
            if not model.is_dir():
                if model.suffix == '.blob':
                    return model, True
                return model, False
            if model_is_blob:
                model_list = get_blob(model)
            else:
                model_list = get_xml(model)
                if not model_list and model_is_blob is None:
                    model_list = get_blob(model)
            if not model_list:
                raise ConfigError('suitable model is not found')
            if len(model_list) != 1:
                raise ConfigError('More than one model matched, please specify explicitly')
            model = model_list[0]
            print_info('Found model {}'.format(model))
            return model, model.suffix == '.blob'

        model, is_blob = get_model()
        if is_blob:
            return model, None
        weights = self.get_value_from_config('weights')
        if weights is None or Path(weights).is_dir():
            weights_dir = weights or model.parent
            weights = Path(weights_dir) / model.name.replace('xml', 'bin')
            print_info('Found weights {}'.format(get_path(weights)))
        return model, weights

    def _is_fpga(self):
        device_list = map(lambda device: device.split('.')[0], self._devices_list())
        return 'FPGA' in device_list

    def _is_vpu(self):
        device_list = map(lambda device: device.split('.')[0], self._devices_list())
        return contains_any(device_list, VPU_PLUGINS)

    def _prepare_bitstream_firmware(self, config):
        if not self._is_fpga():
            return

        compiler_mode = os.environ.get(FPGA_COMPILER_MODE_VAR)
        if compiler_mode == '3':
            return

        bitstream = config.get('bitstream')
        if bitstream:
            previous_bitstream = config.get('_prev_bitstream', '')
            if str(previous_bitstream) != str(bitstream):
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
            else:
                os.environ[FPGA_COMPILER_MODE_VAR] = '3'
                self._set_variable = True

    @staticmethod
    def get_cpu_extension(cpu_extensions, selection_mode):
        def get_cpu_extensions_list(file_format, base_name, selection_mode):
            if not selection_mode:
                default_cpu_extension = file_format.format(base_name)
                extension_list = list(extensions_path.glob(default_cpu_extension))

                if extension_list:
                    return extension_list

                if get_cpu_info is None:
                    raise ValueError('CPU extensions automatic search requires pycpuinfo. '
                                     'Please install it or set cpu extensions lib directly')

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
            'Linux': ('lib{}.so',),
            'Windows': ('{}.dll',),
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
    def convert_model(config, framework=None):
        if framework is None:
            framework = DLSDKLauncherConfigValidator.check_model_source(config)
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
            get_parameter_value_from_config(
                config, DLSDKLauncher.parameters(), '_tf_obj_detection_api_pipeline_config_path'
            ),
            get_parameter_value_from_config(
                config, DLSDKLauncher.parameters(), '_transformations_config_dir'
            ),
            should_log_cmd=should_log_mo_cmd
        )

    @property
    def num_requests(self):
        return self._num_requests

    @num_requests.setter
    def num_requests(self, num_ireq: int):
        if num_ireq != self._num_requests:
            self._num_requests = num_ireq
            self.load_network(self.network, log=False)

    @property
    def async_mode(self):
        return self._async_mode

    @async_mode.setter
    def async_mode(self, flag):
        if flag:
            if 'CPU' in self._devices_list():
                self.ie_core.set_config({'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}, 'CPU')
            if 'GPU' in self._devices_list():
                self.ie_core.set_config({'GPU_THROUGHPUT_STREAMS': 'GPU_THROUGHPUT_AUTO'}, 'GPU')
        self._async_mode = flag

    def get_async_requests(self):
        return [AsyncInferRequestWrapper(ireq_id, ireq) for ireq_id, ireq in enumerate(self.exec_network.requests)]

    def _reshape_input(self, shapes):
        del self.exec_network
        self.network.reshape(shapes)

        self.exec_network = self.ie_core.load_network(self.network, self.device, num_requests=self._num_requests)

    def _set_batch_size(self, batch_size):
        # in some cases we can not use explicit property for setting batch size, so we need to use reshape instead
        # save const inputs without changes
        has_info = hasattr(self.network, 'input_info')
        if not has_info:
            ie_input_info = self.network.inputs
        else:
            ie_input_info = OrderedDict([(name, data.input_data) for name, data in self.network.input_info.items()])
        const_inputs_shapes = {
            input_name: ie_input_info[input_name].shape for input_name in self.const_inputs
        }
        new_non_const_input_shapes = {}
        for layer_name, layer in ie_input_info.items():
            if layer_name in const_inputs_shapes:
                continue
            layer_shape = layer.shape
            ind_batch = layer.layout.find('N')
            if ind_batch != -1:
                layer_shape[ind_batch] = batch_size
            new_non_const_input_shapes[layer_name] = layer_shape

        self.network.reshape({**const_inputs_shapes, **new_non_const_input_shapes})

    def _align_data_shape(self, data, input_blob, data_layout):
        input_shape = self.inputs[input_blob].shape
        data_batch_size = data.shape[0]
        input_batch_size = input_shape[0]

        if data_batch_size < input_batch_size:
            warning_message = 'data batch {} is not equal model input batch_size {}.'.format(
                data_batch_size, input_batch_size
            )
            warning(warning_message)
            diff_number = input_batch_size - data_batch_size
            filled_part = [data[-1]] * diff_number
            data = np.concatenate([data, filled_part])
        precision = self.inputs[input_blob].precision
        data = data.astype(PRECISION_TO_DTYPE[precision])
        data_layout = DIM_IDS_TO_LAYOUT.get(tuple(data_layout))
        input_layout = self.inputs[input_blob].layout
        layout_mismatch = (
            data_layout is not None and len(input_layout) == len(data_layout) and input_layout != data_layout
        )
        if layout_mismatch and Blob is not None and self.network is None:
            self._target_layout_mapping[input_blob] = data_layout
            self._use_set_blob = True
            return data

        return data.reshape(input_shape)

    def _prepare_ie(self, log=True):
        if log:
            print_info('IE version: {}'.format(ie.get_version()))
        if self._is_multi():
            self._prepare_multi_device(log)
        else:
            self.async_mode = self.get_value_from_config('async_mode')
            self._set_nireq()

            if log:
                self._log_versions()
        self._device_specific_configuration()

    def _device_specific_configuration(self):
        cpu_extensions = self.config.get('cpu_extensions')
        if 'CPU' in self._devices_list():
            if cpu_extensions:
                selection_mode = self.config.get('_cpu_extensions_mode')
                cpu_extensions = DLSDKLauncher.get_cpu_extension(cpu_extensions, selection_mode)
                self.ie_core.add_extension(str(cpu_extensions), 'CPU')
            self.ie_core.set_config({'CPU_BIND_THREAD': 'YES' if not self._is_multi() else 'NO'}, 'CPU')
        gpu_extensions = self.config.get('gpu_extensions')
        if 'GPU' in self._devices_list():
            config = {}
            if gpu_extensions:
                config['CONFIG_FILE'] = str(gpu_extensions)
            if self._is_multi() and 'CPU' in self._devices_list():
                config['CLDNN_PLUGIN_THROTTLE'] = '1'
            if config:
                self.ie_core.set_config(config, 'GPU')
        if self._is_vpu():
            device_list = map(lambda device: device.split('.')[0], self._devices_list())
            devices = [vpu_device for vpu_device in VPU_PLUGINS if vpu_device in device_list]
            log_level = self.config.get('_vpu_log_level')
            if log_level:
                for device in devices:
                    self.ie_core.set_config({'LOG_LEVEL': log_level}, device)
        device_config = self.config.get('_device_config')
        if device_config:
            self._set_device_config(device_config)

    def _set_nireq(self):
        num_requests = self.config.get('num_requests')
        if num_requests is not None and num_requests != 'AUTO':
            num_requests = get_or_parse_value(num_requests, casting_type=int)
            if len(num_requests) != 1:
                raise ConfigError('Several values for _num_requests specified')
            self._num_requests = num_requests[0]
            if self._num_requests != 1 and not self.async_mode:
                warning('{} infer requests in sync mode is not supported. Only 1 infer request will be used.')
                self._num_requests = 1
        elif not self.async_mode:
            self._num_requests = 1
        else:
            self._num_requests = self.auto_num_requests()
        if self.async_mode:
            print_info('Async mode activated')
            print_info('Infer requests number:{}'.format(self.num_requests))

    def auto_num_requests(self, return_list=False):
        concurrency_device = {
            'CPU': 1,
            'GPU': 1,
            'HDDL': 100,
            'MYRIAD': 4,
            'FPGA': 3
        }
        platform_list = self._devices_list()
        if 'CPU' in platform_list and len(platform_list) == 1:
            min_requests = [4, 5, 3]
            cpu_count = multiprocessing.cpu_count()
            for min_request in min_requests:
                if cpu_count % min_request == 0:
                    num_req = max(min_request, cpu_count / min_request)
                    return num_req if not return_list else [num_req]
        if 'GPU' in platform_list and len(platform_list) == 1:
            return 2 if not return_list else [2]
        per_device_requests = []
        for device in platform_list:
            per_device_requests.append(concurrency_device.get(device, 1))

        return per_device_requests if return_list else sum(per_device_requests)

    def _prepare_multi_device(self, log=True):
        async_mode = self.get_value_from_config('async_mode')
        if not async_mode:
            warning('Using multi device in sync mode non-applicable. Async mode will be used.')
        num_per_device_req = re.findall(NIREQ_REGEX, self._device)
        device_list = self._devices_list()
        num_devices = len(device_list)
        if num_per_device_req:
            brackets = r"(\()|(\))"
            num_per_device_requests = [int(re.sub(brackets, '', nreq)) for nreq in num_per_device_req]
            if 'num_requests' in self.config:
                warning(
                    "number requests already provided in device name specification. "
                    "'num_requests' option will be ignored."
                )
        elif 'num_requests' in self.config and self.config['num_requests'] != 'AUTO':
            num_per_device_requests = get_or_parse_value(self.config['num_request'], casting_type=int)
        else:
            num_per_device_requests = self.auto_num_requests(return_list=True)

        if len(num_per_device_requests) == 1:
            num_per_device_requests = [num_per_device_requests[0]] * len(device_list)

        if num_devices != len(num_per_device_requests):
            raise ConfigError('num requests for all {} should be specified'.format(num_devices))
        self._num_requests = sum(num_per_device_requests) * 2
        if log:
            self._log_versions()
            print_info('Async mode activated')
            print_info('Request number for each device:')
            for device, nreq in zip(device_list, num_per_device_requests):
                print_info('    {} - {}'.format(device, nreq))

    def _set_device_config(self, device_config):
        device_specific_configuration = read_yaml(device_config, ordered=False)
        if not isinstance(device_specific_configuration, dict):
            raise ConfigError('device configuration should be a dict-like')
        self.ie_core.set_config(device_specific_configuration, self.device)

    def _log_versions(self):
        versions = self.ie_core.get_versions(self._device)
        print_info("Loaded {} plugin version:".format(self._device))
        for device_name, device_version in versions.items():
            print_info("    {device_name} - {descr}: {maj}.{min}.{num}".format(
                device_name=device_name, descr=device_version.description, maj=device_version.major,
                min=device_version.minor, num=device_version.build_number
            ))

    def _create_network(self, input_shapes=None):
        model_path = Path(self._model)
        compiled_model = model_path.suffix == '.blob'
        if compiled_model:
            self.network = None
            self.exec_network = self.ie_core.import_network(str(self._model), self._device)
            self.original_outputs = list(self.exec_network.outputs.keys())
            has_info = hasattr(self.exec_network, 'input_info')
            if has_info:
                ie_input_info = OrderedDict([
                    (name, data.input_data) for name, data in self.exec_network.input_info.items()
                ])
            else:
                ie_input_info = self.exec_network.inputs
            first_input = next(iter(ie_input_info))
            input_info = ie_input_info[first_input]
            batch_pos = input_info.layout.find('N')
            self._batch = input_info.shape[batch_pos] if batch_pos != -1 else 1
            return
        if self._weights is None:
            self._weights = model_path.parent / (model_path.name.split(model_path.suffix)[0] + '.bin')
        self.network = self.read_network(self._model, self._weights)
        self.original_outputs = self.network.outputs
        outputs = self.config.get('outputs')
        if outputs:
            def output_preprocessing(output_string):
                output_tuple = string_to_tuple(output_string, casting_type=None)
                if len(output_tuple) == 1:
                    return output_string
                return tuple([output_tuple[0], int(output_tuple[1])])

            preprocessed_outputs = [output_preprocessing(output) for output in outputs]
            self.network.add_outputs(preprocessed_outputs)

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

    def load_network(self, network=None, log=False):
        if hasattr(self, 'exec_network'):
            del self.exec_network
        if network is None:
            self._create_network()
        else:
            self.network = network
        self._set_precision()
        if log:
            self._print_input_output_info()

        if self.network:
            self.exec_network = self.ie_core.load_network(self.network, self._device, num_requests=self.num_requests)

    def load_ir(self, xml_path, bin_path, log=False):
        self._model = xml_path
        self._weights = bin_path
        self.load_network(log=log)

    def read_network(self, model, weights):
        if 'read_network' in ie.IECore.__dict__:
            network = self.ie_core.read_network(model=str(model), weights=str(weights))
        else:
            network = ie.IENetwork(model=str(model), weights=str(weights))

        return network

    def inputs_info_for_meta(self):
        return {
            layer_name: layer.shape for layer_name, layer in self.inputs.items()
            if layer_name not in self.const_inputs + self.image_info_inputs
        }

    def fit_to_input(self, data, layer_name, layout, precision):
        def data_to_blob(layer_shape, data):
            data_shape = np.shape(data)
            if len(layer_shape) == 4:
                if len(data_shape) == 5:
                    data = data[0]

                if len(data_shape) < 4:
                    if len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape))):
                        return np.resize(data, layer_shape)
                return np.transpose(data, layout)

            if len(layer_shape) == 2:
                if len(data_shape) == 1:
                    return np.transpose([data])
                if len(layout) == 2:
                    return np.transpose(data, layout)

            if len(layer_shape) == 5 and len(layout) == 5:
                return np.transpose(data, layout)

            return np.array(data)

        layer_shape = tuple(self.inputs[layer_name].shape)

        data = data_to_blob(layer_shape, data)
        if precision:
            data = data.astype(precision)

        data_shape = np.shape(data)
        if data_shape != layer_shape:
            if self.allow_reshape_input:
                self._do_reshape = True
                return data

        return self._align_data_shape(data, layer_name, layout)

    def _set_precision(self):
        has_info = hasattr(self.network if self.network is not None else self.exec_network, 'input_info')
        config_inputs = self.config.get('inputs', [])
        for input_config in config_inputs:
            if 'precision' in input_config:
                if self.network:
                    if not has_info:
                        self.network.inputs[input_config['name']].precision = input_config['precision']
                    else:
                        self.network.input_info[input_config['name']].precision = input_config['precision']
                else:
                    if not has_info:
                        self.exec_network.inputs[input_config['name']].precision = input_config['precision']
                    else:
                        self.exec_network.input_info[input_config['name']].precision = input_config['precision']

    def _print_input_output_info(self):
        print_info('Input info:')
        has_info = hasattr(self.network if self.network is not None else self.exec_network, 'input_info')
        if self.network:
            if has_info:
                network_inputs = OrderedDict(
                    [(name, data.input_data) for name, data in self.network.input_info.items()]
                )
            else:
                network_inputs = self.network.inputs
            network_outputs = self.network.outputs
        else:
            if has_info:
                network_inputs = OrderedDict([
                    (name, data.input_data) for name, data in self.exec_network.input_info.items()
                ])
            else:
                network_inputs = self.exec_network.inputs
            network_outputs = self.exec_network.outputs
        for name, input_info in network_inputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(input_info.precision))
            print_info('\tshape {}\n'.format(input_info.shape))
        print_info('Output info')
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            print_info('\tshape: {}\n'.format(output_info.shape))

    def release(self):
        if 'network' in self.__dict__:
            del self.network
        if 'exec_network' in self.__dict__:
            del self.exec_network
        if 'ie_core' in self.__dict__:
            del self.ie_core
        if self._set_variable:
            del os.environ[FPGA_COMPILER_MODE_VAR]
