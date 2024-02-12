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

import multiprocessing
from pathlib import Path
import re
import warnings
from collections import OrderedDict
import numpy as np
import openvino.inference_engine as ie  # pylint: disable=package-absolute-imports

from .dlsdk_launcher_config import (
    HETERO_KEYWORD, MULTI_DEVICE_KEYWORD, NIREQ_REGEX, VPU_PLUGINS,
    get_cpu_extension,
    DLSDK_LAUNCHER_PARAMETERS,
    DLSDKLauncherConfigValidator,
    automatic_model_search
)
from .dlsdk_async_request import AsyncInferRequestWrapper
from ..config import ConfigError
from ..logging import warning
from ..utils import (
    read_yaml,
    contains_any,
    string_to_tuple,
    get_or_parse_value,
    UnsupportedPackage,
    parse_partial_shape
)
from .launcher import Launcher
from ..logging import print_info
from .input_feeder import PRECISION_TO_DTYPE, DIM_IDS_TO_LAYOUT

try:
    from openvino.inference_engine import Blob, TensorDesc  # pylint: disable=import-outside-toplevel,package-absolute-imports
except ImportError:
    try:
        # old structures names compatibilities
        from openvino.inference_engine import IEBlob, IETensorDesc  # pylint: disable=import-outside-toplevel,package-absolute-imports

        Blob = IEBlob
        TensorDesc = IETensorDesc
    except ImportError:
        Blob, TensorDesc = None, None

try:
    import ngraph as ng
except ImportError as error:
    ng = UnsupportedPackage('ngraph', error)


class DLSDKLauncher(Launcher):
    """
    Class for infer model using DLSDK framework.
    """

    __provider__ = 'dlsdk'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(DLSDK_LAUNCHER_PARAMETERS)

        return parameters

    def __init__(self, config_entry, model_name='', delayed_model_loading=False,
                 preprocessor=None, postpone_inputs_configuration=False):
        super().__init__(config_entry, model_name=model_name)
        if ie.get_version().split('-')[0] >= '2022.1.0':
            warnings.warn('dlsdk launcher is deprecated. Please use openvino instead', DeprecationWarning)

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
        self.dynamic_shapes_policy = self.get_value_from_config('_undefined_shapes_resolving_policy')
        self._set_variable = False
        self._async_mode = False
        self._prepare_ie()
        self._delayed_model_loading = delayed_model_loading
        self._postpone_input_configuration = postpone_inputs_configuration
        self._preprocess_info = {}
        self._preprocess_steps = []
        self.disable_resize_to_input = False
        self._do_reshape = False
        self._use_set_blob = False
        self._output_layouts = {}
        self._output_precisions = {}
        self.dyn_input_layers = []
        self._partial_shapes = {}
        self.is_dynamic = False
        self.preprocessor = preprocessor

        if not delayed_model_loading:
            self._model, self._weights = automatic_model_search(
                self._model_name, self.get_value_from_config('model'),
                self.get_value_from_config('weights'),
                self.get_value_from_config('_model_type')
            )
            self.load_network(log=True, preprocessing=preprocessor)
            self.allow_reshape_input = self.get_value_from_config('allow_reshape_input') and self.network is not None
        else:
            self.allow_reshape_input = self.get_value_from_config('allow_reshape_input')
        self._target_layout_mapping = {}
        self._lstm_inputs = None
        if '_list_lstm_inputs' in self.config:
            self._configure_lstm_inputs()
        self.reset_memory_state = self.get_value_from_config('reset_memory_state')

    @classmethod
    def validate_config(cls, config, delayed_model_loading=False, fetch_only=False, uri_prefix=''):
        field_uri = uri_prefix or 'launcher.{}'.format(cls.__provider__)
        return DLSDKLauncherConfigValidator(
            field_uri, fields=cls.parameters(), delayed_model_loading=delayed_model_loading).validate(
                config, field_uri=field_uri, validation_scheme=cls.validation_scheme(), fetch_only=fetch_only)

    @property
    def device(self):
        return self._device

    @property
    def lstm_inputs(self):
        return self._lstm_inputs

    @property
    def inputs(self):
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
        if hasattr(self, 'original_outputs'):
            return next(iter(self.original_outputs))
        return None

    def predict(self, inputs, metadata=None, **kwargs):
        if self._lstm_inputs:
            return self._predict_sequential(inputs, metadata)

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
                    tensor_desc = TensorDesc(ie_input_info[key].precision, input_data.shape, layout)
                    preprocess_info = self._preprocess_info.get(key)
                    if preprocess_info is not None:
                        self.exec_network.requests[0].set_blob(key, Blob(tensor_desc, input_data), preprocess_info)
                    else:
                        self.exec_network.requests[0].set_blob(key, Blob(tensor_desc, input_data))
            result = self.exec_network.infer(infer_inputs) if not self._use_set_blob else self.exec_network.infer()
            results.append(result)
        if self.reset_memory_state:
            for state in self.exec_network.requests[0].query_state():
                state.reset()

        if metadata is not None:
            self._fill_meta(metadata, None if not self.dyn_input_layers else inputs[-1])
        self._do_reshape = False
        self._use_set_blob = self.disable_resize_to_input

        return results

    def _predict_sequential(self, inputs, metadata=None, **kwargs):
        lstm_inputs_feed = self._fill_lstm_inputs()
        results = []
        for feed_dict in inputs:
            feed_dict.update(lstm_inputs_feed)
            output_result = self.exec_network.infer(feed_dict)
            lstm_inputs_feed = self._fill_lstm_inputs(output_result)
            results.append(output_result)

            if self._do_reshape:
                input_shapes = {layer_name: data.shape for layer_name, data in feed_dict.items()}
                self._reshape_input(input_shapes)

        if metadata is not None:
            self._fill_meta(metadata, None if not self.dyn_input_layers else inputs[-1])
        self._do_reshape = False
        return results

    def predict_async(self, ir, inputs, metadata=None, context=None, **kwargs):
        infer_inputs = inputs[0]
        if metadata is not None:
            self._fill_meta(metadata, None if not self.dyn_input_layers else infer_inputs)
        ir.infer(infer_inputs, metadata, context)

    def _fill_meta(self, metadata, inputs=None):
        for meta_ in metadata:
            meta_['input_shape'] = self.inputs_info_for_meta(inputs)
            if self._output_layouts:
                meta_['output_layout'] = self._output_layouts
            if self._output_precisions:
                meta_['output_precision'] = self._output_precisions

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
        if hasattr(self.network, 'layers'):
            self._set_affinity_via_layers(custom_affinity, automatic_affinity)
            return
        if isinstance(ng, UnsupportedPackage):
            ng.raise_error('affinity setting')
        self._set_affinity_ng(custom_affinity, automatic_affinity)

    def _set_affinity_ng(self, custom_affinity, auto_affinity):
        ng_function = ng.function_from_cnn(self.network)
        for node in ng_function.get_ordered_ops():
            layer_name = node.get_friendly_name()
            device = custom_affinity.get(layer_name, auto_affinity.get(layer_name))
            if device is None:
                continue
            if not (device in self._devices_list() or device == self._device):
                raise ConfigError(
                    'Device \'{device}\' set for \'{layer}\' layer is not present in '
                    'provided configuration \'{configuration}\''.format(
                        device=device, layer=layer_name, configuration=self._device
                    )
                )
            rt_info = node.get_rt_info()
            rt_info["affinity"] = device

    def _set_affinity_via_layers(self, custom_affinity, automatic_affinity):
        layers = self.network.layers
        for layer_name in layers:
            device = custom_affinity.get(layer_name, automatic_affinity.get(layer_name))
            if device is None:
                continue
            if device not in self._devices_list():
                raise ConfigError(
                    'Device \'{device}\' set for \'{layer}\' layer is not present in '
                    'provided configuration \'{configuration}\''.format(
                        device=device, layer=layer_name, configuration=self._device
                    )
                )
            layers[layer_name].affinity = device

    def _is_vpu(self):
        device_list = (device.split('.')[0] for device in self._devices_list())
        return contains_any(device_list, VPU_PLUGINS)

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

    def _reshape_input(self, shapes, make_dynamic=False):
        if hasattr(self, 'exec_network'):
            del self.exec_network
        self.network.reshape(shapes)
        self.dyn_input_layers, self._partial_shapes = self.get_dynamic_inputs(self.network)
        if self.dyn_input_layers and make_dynamic:
            return
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
            layer_shape = (
                layer.shape if layer_name not in self._partial_shapes else list(self._partial_shapes[layer_name])
            )
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
        if data_layout is not None:
            data_layout = DIM_IDS_TO_LAYOUT.get(tuple(data_layout))
        input_layout = self.inputs[input_blob].layout
        layout_mismatch = (
            data_layout is not None and len(input_layout) == len(data_layout) and input_layout != data_layout
        )
        if layout_mismatch and Blob is not None and self.network is None or input_blob in self._preprocess_info:
            self._target_layout_mapping[input_blob] = data_layout
            self._use_set_blob = True
            return data
        return data.reshape(input_shape) if not self.disable_resize_to_input else data

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
                cpu_extensions = get_cpu_extension(cpu_extensions, selection_mode)
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
            device_list = (device.split('.')[0] for device in self._devices_list())
            devices = [vpu_device for vpu_device in VPU_PLUGINS if vpu_device in device_list]
            log_level = self.config.get('_vpu_log_level')
            if log_level:
                for device in devices:
                    self.ie_core.set_config({'LOG_LEVEL': log_level}, device)
        device_config = self.config.get('device_config')
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
        concurrency_device = {'CPU': 1, 'GPU': 1, 'HDDL': 100, 'MYRIAD': 4}
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
        if not isinstance(device_config, dict):
            raise ConfigError('device configuration should be a dict-like')
        if all(not isinstance(value, dict) for value in device_config.values()):
            self.ie_core.set_config(dict(device_config), self.device)
        else:
            for key, value in device_config.items():
                if isinstance(value, dict):
                    if key in self._devices_list():
                        if key not in self.ie_core.available_devices:
                            warnings.warn('{} device is unknown. Config loading may lead to error.'.format(key))
                        self.ie_core.set_config(dict(value), key)
                    else:
                        warnings.warn(
                            f'Configuration for {key} will be skipped as device is not listed in evaluation device'
                        )
                else:
                    warnings.warn('Option {key}: {value} will be skipped because device to which it should be '
                                  'applied is not specified or option is not a dict-like'.format(key=key, value=value))

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
        if self._weights is None and self._model.suffix != '.onnx':
            self._weights = model_path.parent / (model_path.name.split(model_path.suffix)[0] + '.bin')
        self.network = self.read_network(self._model, self._weights)
        self.original_outputs = self.network.outputs
        outputs = self.config.get('outputs')
        if outputs:
            def output_preprocessing(output_string):
                output_tuple = string_to_tuple(output_string, casting_type=None)
                if len(output_tuple) == 1:
                    return output_string
                return output_tuple[0], int(output_tuple[1])

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

    def load_network(self, network=None, log=False, preprocessing=None):
        if hasattr(self, 'exec_network'):
            del self.exec_network
        if network is None:
            self._create_network()
        else:
            self.network = network
        if self.network is not None:
            self.dyn_input_layers, self._partial_shapes = self.get_dynamic_inputs(self.network)

        if not self._postpone_input_configuration:
            self._set_precision()
            self._set_input_shape()
            self.dyn_input_layers, self._partial_shapes = self.get_dynamic_inputs(self.network)
            if log:
                self._print_input_output_info()
            if preprocessing:
                self._set_preprocess(preprocessing)
            if self.network and not preprocessing and (not self.dyn_input_layers or self.is_dynamic):
                self.exec_network = self.ie_core.load_network(
                    self.network, self._device, num_requests=self.num_requests
                )

    def update_input_configuration(self, input_config):
        self.config['inputs'] = input_config
        self._set_precision()
        self._set_input_shape()
        self.dyn_input_layers, self._partial_shapes = self.get_dynamic_inputs(self.network)
        self._print_input_output_info()
        if self.preprocessor:
            self._set_preprocess(self.preprocessor)
        if self.network:
            self.exec_network = self.ie_core.load_network(
                self.network, self._device, num_requests=self.num_requests
            )

    @staticmethod
    def get_dynamic_inputs(network):
        def is_dynamic(data_info):
            if hasattr(data_info, 'is_dynamic'):
                return data_info.is_dynamic
            return -1 in data_info.shape or not data_info.shape

        inputs_with_undefined_shapes = []
        outputs_with_undefined_shapes = []
        partial_shapes = {}
        if network is None:
            return inputs_with_undefined_shapes, partial_shapes

        for input_name, input_info in network.input_info.items():
            if is_dynamic(input_info.input_data):
                inputs_with_undefined_shapes.append(input_name)
        for out_name, out_info in network.outputs.items():
            if is_dynamic(out_info):
                outputs_with_undefined_shapes.append(out_name)
        if (inputs_with_undefined_shapes or outputs_with_undefined_shapes) and not isinstance(ng, UnsupportedPackage):
            if hasattr(ng, 'partial_shape_from_data'):
                for input_name in inputs_with_undefined_shapes:
                    partial_shapes[input_name] = parse_partial_shape(ng.partial_shape_from_data(
                        network.input_info[input_name].input_data
                    ))
                for out_name in outputs_with_undefined_shapes:
                    partial_shapes[out_name] = parse_partial_shape(
                        ng.partial_shape_from_data(network.outputs[out_name])
                    )

            else:
                ng_function = ng.function_from_cnn(network)
                for node in ng_function.get_ordered_ops():
                    node_name = node.get_friendly_name()
                    if node_name not in inputs_with_undefined_shapes:
                        continue
                    partial_shapes[node_name] = node.get_partial_shape()

        return inputs_with_undefined_shapes, partial_shapes

    @property
    def dyn_batch_only(self):
        if not self.dyn_input_layers:
            return True
        for input_name in self.dyn_input_layers:
            partial_shape = self._partial_shapes[input_name]
            layout = self.inputs[input_name].layout
            for dim, layout_dim in zip(partial_shape, layout):
                if dim == -1 and layout_dim != 'N':
                    return False
        return True

    def load_ir(self, xml_path, bin_path, log=False):
        self._model = xml_path
        self._weights = bin_path
        self.load_network(log=log)

    def read_network(self, model, weights):
        if 'read_network' in ie.IECore.__dict__:
            if weights is None:
                network = self.ie_core.read_network(model=str(model))
            else:
                network = self.ie_core.read_network(model=str(model), weights=str(weights))
        else:
            network = ie.IENetwork(model=str(model), weights=str(weights))
        return network

    def inputs_info_for_meta(self, inputs=None):
        if inputs:
            return {layer_name: np.shape(data) for layer_name, data in inputs.items()}
        if not self.dyn_input_layers:
            return {
                layer_name: layer.shape for layer_name, layer in self.inputs.items()
                if layer_name not in self.const_inputs + self.image_info_inputs
            }
        input_shapes = {}
        for layer_name, layer in self.inputs.items():
            if layer_name in self.const_inputs + self.image_info_inputs:
                continue
            input_shapes[layer_name] = (
                layer.shape if layer_name not in self.dyn_input_layers else self._partial_shapes.get(layer_name, [])
            )
        return input_shapes

    def initialize_undefined_shapes(self, input_data, template_shapes=None):
        if self.dynamic_shapes_policy in ['default', 'dynamic']:
            try:
                if template_shapes:
                    input_shapes = {
                        layer_name: template_shapes.get(layer_name, data.shape) for layer_name, data in
                        input_data[0].items()
                    }
                    self._reshape_input(input_shapes)
                    self.load_network(self.network)
                    self.is_dynamic = True
                if not hasattr(self, 'exec_network') or self.exec_network is None:
                    self.is_dynamic = True
                    self.load_network(self.network)
                self.exec_network.infer(input_data[0])
                return
            except RuntimeError as e:
                if self.dynamic_shapes_policy == 'dynamic':
                    raise e
                self.is_dynamic = False
        input_shapes = {layer_name: data.shape for layer_name, data in input_data[0].items()}
        self._reshape_input(input_shapes)

    def resolve_undefined_batch(self):
        if self.dynamic_shapes_policy in ['default', 'dynamic']:
            try:
                self.is_dynamic = True
                self.load_network(self.network)
            except RuntimeError as e:
                if self.dynamic_shapes_policy == 'dynamic':
                    raise e
                self.is_dynamic = False
        if not self.is_dynamic:
            self._set_batch_size(self._batch)
            self.load_network(self.network)

    def fit_to_input(self, data, layer_name, layout, precision, template=None):
        if layer_name in self.dyn_input_layers:
            layer_rang = len(self._partial_shapes[layer_name])
            input_template = template.get(layer_name) if template else template
            data, l_template = self._data_to_blob_dyn(layer_rang, data, layout, input_template)
            layer_shape = data.shape
            if l_template is not None:
                template[layer_name] = l_template
        else:
            layer_shape = tuple(self.inputs[layer_name].shape)
            data = self._data_to_blob(layer_shape, data, layout)
        if precision:
            data = data.astype(precision)
        if layer_name in self.dyn_input_layers:
            self._do_reshape = not self.is_dynamic
            return data, template
        data_shape = np.shape(data)
        if data_shape != layer_shape:
            if self.allow_reshape_input:
                self._do_reshape = True
                return data
        return self._align_data_shape(data, layer_name, layout)

    @staticmethod
    def _data_to_blob_dyn(layer_rang, data, layout, template=None):
        data_shape = np.shape(data)
        if len(data_shape) - layer_rang == 1 and data_shape[0] == 1:
            if len(data_shape) == len(layout):
                data = np.transpose(data, layout)
                if template is not None and len(template) == layer_rang:
                    tmp_template = [1, ] + template
                    new_template = [tmp_template[l_dim] for l_dim in layout][1:]
                    template = new_template
            data = data[0]
            data_shape = np.shape(data)
        if template is not None:
            if len(template) < np.ndim(data):
                template = [1] * (np.ndim(data) - len(template)) + template
            if len(template) > np.ndim(data):
                template = template[0]
        if len(layout) == len(data_shape):
            if template is not None:
                new_template = [template[l_dim] for l_dim in layout]
                template = new_template
            return np.transpose(data, layout), template
        return np.array(data), template

    def _data_to_blob(self, layer_shape, data, layout):  # pylint:disable=R0911,R0912
        data_shape = np.shape(data)
        if len(layer_shape) == 4:
            if len(data_shape) == 5:
                data = data[0]
            if len(data_shape) == 3:
                data = np.expand_dims(data, -1)
            data_shape = np.shape(data)
            if len(data_shape) < 4:
                if len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape))):
                    return np.resize(data, layer_shape)
            return np.transpose(data, layout) if layout is not None else data
        if len(layer_shape) == 2:
            if len(data_shape) == 1:
                return np.transpose([data])
            if len(data_shape) > 2:
                if all(dim == 1 for dim in layer_shape) and all(dim == 1 for dim in data_shape):
                    return np.resize(data, layer_shape)
                if len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape))):
                    return np.resize(data, layer_shape)
        if len(layer_shape) == 3 and len(data_shape) == 4:
            return np.transpose(data, layout)[0] if layout is not None else data[0]
        if len(layer_shape) == 1:
            return np.resize(data, layer_shape)
        if (len(data_shape) == 3) and (len(layer_shape) == 2) and (data_shape[0] == 1) and (
                data_shape[1] == 1) and self.allow_reshape_input:
            return data[0]
        if layout is not None and len(layer_shape) == len(layout):
            return np.transpose(data, layout)
        if (
                len(layer_shape) == 1 and len(data_shape) > 1 and
                len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape)))
        ):
            return np.resize(data, layer_shape)
        return np.array(data)

    def _set_precision(self):
        has_info = hasattr(self.network if self.network is not None else self.exec_network, 'input_info')
        config_inputs = self.config.get('inputs', [])
        for input_config in config_inputs:
            if 'precision' in input_config:
                if self.network:
                    if not has_info:
                        self.network.inputs[input_config['name']].precision = input_config['precision'].upper()
                    else:
                        self.network.input_info[input_config['name']].precision = input_config['precision'].upper()

    def _set_input_shape(self):
        if not self.network:
            return
        config_inputs = self.config.get('inputs', [])
        input_shapes = {}
        make_dynamic = False
        for input_config in config_inputs:
            if 'shape' in input_config:
                input_shapes[input_config['name']] = input_config['shape']
                if -1 in input_config['shape']:
                    make_dynamic = True
        if not input_shapes:
            return
        orig_input_shapes = {
            input_name: input_info.shape
            if input_name not in self._partial_shapes else self._partial_shapes[input_name]
            for input_name, input_info in self.inputs.items()
        }
        orig_input_shapes.update(input_shapes)
        self._reshape_input(orig_input_shapes, make_dynamic)

    def _configure_lstm_inputs(self):
        lstm_mapping = {}
        config_inputs = self.config.get('inputs', [])
        for input_config in config_inputs:
            if input_config['type'] == 'LSTM_INPUT':
                lstm_mapping[input_config['name']] = input_config['value']
        self._lstm_inputs = lstm_mapping

    def _fill_lstm_inputs(self, infer_outputs=None):
        feed_dict = {}
        for lstm_var, output_layer in self._lstm_inputs.items():
            layer_shape = self.inputs[lstm_var].shape
            input_data = infer_outputs[output_layer].reshape(layer_shape) if infer_outputs else np.zeros(
                layer_shape, dtype=PRECISION_TO_DTYPE[self.inputs[lstm_var].precision]
            )
            feed_dict[lstm_var] = input_data
        return feed_dict

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
            print_info(
                '\tshape: {}\n'.format(
                    input_info.shape if name not in self.dyn_input_layers else self._partial_shapes.get(name, [])
                )
            )
        print_info('Output info')
        for name, output_info in network_outputs.items():
            print_info('\tLayer name: {}'.format(name))
            print_info('\tprecision: {}'.format(output_info.precision))
            if not hasattr(output_info, 'is_dynamic'):
                shape = output_info.shape
            else:
                if output_info.is_dynamic:
                    shape = self._partial_shapes.get(name, [])
                else:
                    shape = output_info.shape
            print_info('\tshape: {}\n'.format(shape))
            self._output_precisions[name] = PRECISION_TO_DTYPE[output_info.precision]
            self._output_layouts[name] = output_info.layout

    def _set_preprocess(self, preprocess):
        if preprocess.ie_processor is None:
            return
        if self.network is not None:
            self.disable_resize_to_input = False
            preprocess_steps = preprocess.ie_preprocess_steps
            if not preprocess_steps:
                return
            for input_name, input_info in self.network.input_info.items():
                if input_name in self.const_inputs + self.image_info_inputs:
                    continue
                for (name, value) in preprocess_steps:
                    setattr(input_info.preprocess_info, name, value)
                if preprocess.ie_processor.has_normalization():
                    channel_id = input_info.layout.find('C')
                    if channel_id != -1:
                        num_channels = input_info.input_data.shape[channel_id]
                        preprocess.ie_processor.set_normalization(num_channels, input_info.preprocess_info)
            self.disable_resize_to_input = preprocess.ie_processor.has_resize()
            self._use_set_blob = self.disable_resize_to_input
            self.load_network(self.network)
            self._preprocess_steps = preprocess_steps
            return
        preprocess_info_by_input = OrderedDict()
        preprocess_info = preprocess.preprocess_info
        for input_name in self.inputs:
            if input_name in self.const_inputs + self.image_info_inputs:
                continue
            if preprocess.ie_processor.has_normalization():
                channel_id = self.inputs[input_name].layout.find('C')
                if channel_id != -1:
                    num_channels = self.inputs[input_name].shape[channel_id]
                    preprocess.ie_processor.set_normalization(num_channels, preprocess_info)
            preprocess_info_by_input[input_name] = preprocess_info
        self._preprocess_info = preprocess_info_by_input
        self.disable_resize_to_input = preprocess.ie_processor.has_resize()

    def get_model_file_type(self):
        if hasattr(self, '_model'):
            return self._model.suffix
        return None

    def input_shape(self, input_name):
        if input_name in self._partial_shapes:
            return self._partial_shapes[input_name]
        return self.inputs[input_name].shape

    def release(self):
        if 'network' in self.__dict__:
            del self.network
        if 'exec_network' in self.__dict__:
            del self.exec_network
        if 'ie_core' in self.__dict__:
            del self.ie_core
