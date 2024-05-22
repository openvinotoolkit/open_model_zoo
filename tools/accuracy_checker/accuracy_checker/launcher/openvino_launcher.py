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
#pylint:disable=no-name-in-module
#pylint:disable=package-absolute-imports
#pylint:disable=too-many-lines
import io
import multiprocessing
from pathlib import Path
import re
import numpy as np
from openvino import Core, AsyncInferQueue, get_version, PartialShape, Type, Dimension
from openvino.preprocess import PrePostProcessor
from .dlsdk_launcher_config import (
    HETERO_KEYWORD, MULTI_DEVICE_KEYWORD, NIREQ_REGEX, VPU_PLUGINS, AUTO_DEVICE_KEYWORD, AUTO_SINGLE_DEVICE_KEYWORD,
    get_cpu_extension,
    DLSDK_LAUNCHER_PARAMETERS,
    DLSDKLauncherConfigValidator,
    automatic_model_search,
    ov_set_config
)
from .dlsdk_async_request import AsyncInferRequestWrapper
from ..config import ConfigError
from ..logging import warning, debug, print_info
from ..utils import (
    read_yaml,
    contains_any,
    string_to_tuple,
    get_or_parse_value,
    parse_partial_shape,
    postprocess_output_name
)
from .launcher import Launcher


format_map = {
      'f32': np.float32, 'i32': np.int32, 'i64': int,
      'fp16': np.float16, 'f16': np.float16, 'i16': np.int16, 'u16': np.uint16,
      'i8': np.int8, 'u8': np.uint8, 'boolean': np.uint8
}

PRECISION_STR_TO_TYPE = {
    'FP32': Type.f32, 'FP16': Type.f16, 'U8': Type.u8, 'U16': Type.u16, 'I8': Type.i8, 'I16': Type.i16,
    'I32': Type.i32, 'I64': Type.i64, 'BOOL': Type.boolean, 'INT8': Type.u8, 'BF16': Type.bf16
}


class OpenVINOLauncher(Launcher):
    __provider__ = 'openvino'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(DLSDK_LAUNCHER_PARAMETERS)
        return parameters

    def __init__(self, config_entry, model_name='', delayed_model_loading=False,
                 preprocessor=None, postpone_inputs_configuration=False):
        super().__init__(config_entry, model_name=model_name)
        self._set_variable = False
        self.ie_config = self.config.get('ie_config')
        self.ie_core = Core()
        if self.ie_config:
            ov_set_config(self.ie_core, self.ie_config)
        self._delayed_model_loading = delayed_model_loading
        dlsdk_launcher_config = DLSDKLauncherConfigValidator(
            'OpenVINO_Launcher', fields=self.parameters(), delayed_model_loading=delayed_model_loading,
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
        self._output_layouts = {}
        self._output_precisions = {}
        self.dyn_input_layers = []
        self._partial_shapes = {}
        self.is_dynamic = False
        self.preprocessor = preprocessor
        self.infer_request = None
        self._num_requests = None

        if not delayed_model_loading:
            self._model, self._weights = automatic_model_search(self._model_name, self.get_value_from_config('model'),
                                                                self.get_value_from_config('weights'),
                                                                self.get_value_from_config('_model_type'))
            self.load_network(log=not postpone_inputs_configuration, preprocessing=preprocessor)
            self.allow_reshape_input = self.get_value_from_config('allow_reshape_input') and self.network is not None
            if not postpone_inputs_configuration:
                self.try_to_set_default_layout()
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

    def try_to_set_default_layout(self):
        if self.get_value_from_config('_model_type') in ['tf', 'tflite']:
            self.default_layout = 'NHWC'
        input_nodes = self.network.inputs if self.network else self.exec_network.inputs
        for input_node in input_nodes:
            shape = parse_partial_shape(input_node.get_node().partial_shape)
            if len(shape) != 4:
                continue
            if input_node.get_node().layout.has_name('C'):
                channel_dim = input_node.get_node().layout.get_index_by_name('C')
                if channel_dim in [3, -1]:
                    self.default_layout = 'NHWC'
                    return
            if shape[-1] in [1, 2, 3, 4, 6, 9]:
                self.default_layout = 'NHWC'
                return
        self.default_layout = 'NCHW'
        return

    @property
    def device(self):
        return self._device

    @property
    def inputs(self):
        inputs = self.exec_network.inputs if self.network is None else self.network.inputs
        return {input_info.get_node().friendly_name: input_info.get_node() for input_info in inputs}

    @property
    def batch(self):
        return self._batch

    @property
    def output_blob(self):
        if hasattr(self, 'original_outputs'):
            return next(iter(self.original_outputs)).get_node().friendly_name
        return None

    @property
    def additional_output_mapping(self):
        if hasattr(self, 'out_tensor_name_to_node'):
            return self.out_tensor_name_to_node
        return {}

    def predict(self, inputs, metadata=None, return_raw=False, **kwargs):
        if self._lstm_inputs:
            return self._predict_sequential(inputs, metadata, return_raw)
        results = []
        raw_results = []
        for infer_inputs in inputs:
            if self._do_reshape:
                input_shapes = {layer_name: data.shape for layer_name, data in infer_inputs.items()}
                self._reshape_input(input_shapes)
            if self.infer_request is None:
                self.infer_request = self.exec_network.create_infer_request()
            feed_dict = {self.input_to_tensor_name[layer_name]: data for layer_name, data in infer_inputs.items()}
            outputs = self.infer_request.infer(inputs=feed_dict)
            raw_results.append(outputs)
            results.append({out_node.get_node().friendly_name: out_res for out_node, out_res in outputs.items()})
        if self.reset_memory_state:
            for state in self.infer_request.query_state():
                state.reset()

        if metadata is not None:
            self._fill_meta(metadata, None if not self.dyn_input_layers else inputs[-1])
        self._do_reshape = False
        if return_raw:
            return results, raw_results
        return results

    def _predict_sequential(self, inputs, metadata=None, return_raw=False, **kwargs):
        lstm_inputs_feed = self._fill_lstm_inputs()
        if not self.infer_request:
            self.infer_request = self.exec_network.create_infer_request()
        results = []
        raw_results = []
        for feed_dict in inputs:
            feed_dict.update(lstm_inputs_feed)
            infer_inputs = {self.input_to_tensor_name[layer_name]: data for layer_name, data in feed_dict.items()}
            out_tensors = self.infer_request.infer(infer_inputs)
            output_result = {
                out_node.get_node().friendly_name: out_tensor for out_node, out_tensor in out_tensors.items()}
            lstm_inputs_feed = self._fill_lstm_inputs(output_result)
            results.append(output_result)
            if return_raw:
                raw_results.append(out_tensors)

            if self._do_reshape:
                input_shapes = {layer_name: data.shape for layer_name, data in feed_dict.items()}
                self._reshape_input(input_shapes)

        if metadata is not None:
            self._fill_meta(metadata, None if not self.dyn_input_layers else inputs[-1])
        self._do_reshape = False
        if return_raw:
            return results, raw_results
        return results

    def predict_async(self, ir, inputs, metadata=None, context=None, **kwargs):
        infer_inputs = inputs[0]
        feed_dict = {self.input_to_tensor_name[name]: data for name, data in infer_inputs.items()}
        if metadata is not None:
            self._fill_meta(metadata, None if not self.dyn_input_layers else infer_inputs)
        ir.infer(feed_dict, metadata, context)

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

    def _is_auto(self):
        return self._device.startswith(AUTO_SINGLE_DEVICE_KEYWORD)

    def _devices_list(self):
        device = self._device
        if self._is_hetero():
            device = self._device[len(HETERO_KEYWORD):]
        if self._is_multi():
            device = self._device[len(MULTI_DEVICE_KEYWORD):]
            device = re.sub(NIREQ_REGEX, '', device)
        if self._is_auto():
            if device == AUTO_SINGLE_DEVICE_KEYWORD:
                return [self._device]
            device = self._device[len(AUTO_DEVICE_KEYWORD):]
        return [platform_.upper().strip() for platform_ in device.split(',')]

    def _set_affinity(self, affinity_map_path):
        auto_affinity = self.ie_core.query_network(self.network, self._device)
        custom_affinity = read_yaml(affinity_map_path)
        for layer in custom_affinity:
            if layer not in auto_affinity:
                raise ConfigError('Layer \'{layer}\' is not present in network'.format(layer=layer))
        for node in self.network.get_ordered_ops():
            layer_name = node.friendly_name
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
            node.rt_info["affinity"] = device

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

    @property
    def async_mode(self):
        return self._async_mode

    @async_mode.setter
    def async_mode(self, flag):
        for device in self._devices_list():
            ov_set_config(
                self.ie_core, {'PERFORMANCE_HINT': 'THROUGHPUT' if flag else 'LATENCY'}, device=device.upper())
        self._async_mode = flag

    def get_async_requests(self):
        self._set_nireq()
        return [AsyncInferRequestWrapper(ireq_id, self.exec_network.create_infer_request())
                for ireq_id in range(self.num_requests)]

    def _reshape_input(self, shapes, make_dynamic=False):
        if hasattr(self, 'exec_network'):
            del self.exec_network
        if self.infer_request is not None:
            del self.infer_request
            self.infer_request = None
        partial_shapes = {}
        for name, shape in shapes.items():
            initial_partial_shape = self.inputs[name].partial_shape
            if len(shape) < int(str(initial_partial_shape.rank)):
                required_shape = list(parse_partial_shape(initial_partial_shape))
                required_shape[-1*len(shape):] = shape
                shape = required_shape
            p_shape = PartialShape(
                [Dimension(d if d != 0 else -1) if not isinstance(d, tuple) else Dimension(d[0], d[1]) for d in shape])
            partial_shapes[self.input_to_index[name]] = p_shape
        self.network.reshape(partial_shapes)
        self.dyn_input_layers, self._partial_shapes = self.get_dynamic_inputs(self.network)
        if self.dyn_input_layers and make_dynamic:
            return
        self.exec_network = self.ie_core.compile_model(self.network, self.device)
        self.infer_request = self.exec_network.create_infer_request()

    @staticmethod
    def reshape_network(network, shapes):
        partial_shapes = {}
        for name, shape in shapes.items():
            p_shape = PartialShape(
                [Dimension(d) if not isinstance(d, tuple) else Dimension(d[0], d[1]) for d in shape])
            partial_shapes[name] = p_shape
        network.reshape(partial_shapes)
        return network

    def _align_data_shape(self, data, input_blob, data_layout):
        input_shape = self.inputs[input_blob].shape
        data_batch_size = data.shape[0]
        input_batch_size = input_shape[0]
        if data_batch_size < input_batch_size:
            warning_message = 'data batch {} is not equal model input batch_size {}.'.format(
                data_batch_size, input_batch_size)
            warning(warning_message)
            diff_number = input_batch_size - data_batch_size
            filled_part = [data[-1]] * diff_number
            data = np.concatenate([data, filled_part])
        return data.reshape(input_shape) if not self.disable_resize_to_input else data

    def _prepare_ie(self, log=True):
        if log:
            print_info('IE version: {}'.format(get_version()))
        if self._is_multi():
            self._prepare_multi_device(log)
        else:
            self.async_mode = self.get_value_from_config('async_mode')
            if log:
                self._log_versions()
        self._device_specific_configuration()

    def _device_specific_configuration(self):
        cpu_extensions = self.config.get('cpu_extensions')
        if 'CPU' in self._devices_list():
            if cpu_extensions:
                selection_mode = self.config.get('_cpu_extensions_mode')
                cpu_extensions = get_cpu_extension(cpu_extensions, selection_mode)
                self.ie_core.add_extension(str(cpu_extensions))
            ov_set_config(
                self.ie_core, {'ENABLE_CPU_PINNING': 'YES' if not self._is_multi() else 'NO'}, device='CPU')
        gpu_extensions = self.config.get('gpu_extensions')
        if 'GPU' in self._devices_list():
            config = {}
            if gpu_extensions:
                config['CONFIG_FILE'] = str(gpu_extensions)
            if self._is_multi() and 'CPU' in self._devices_list():
                config['CLDNN_PLUGIN_THROTTLE'] = '1'
            if config:
                ov_set_config(self.ie_core, config, device='GPU')
        if self._is_vpu():
            device_list = (device.split('.')[0] for device in self._devices_list())
            devices = [vpu_device for vpu_device in VPU_PLUGINS if vpu_device in device_list]
            log_level = self.config.get('_vpu_log_level')
            if log_level:
                for device in devices:
                    ov_set_config(self.ie_core, {'LOG_LEVEL': log_level}, device=device)
        device_config = self.config.get('device_config')
        if device_config:
            self._set_device_config(device_config)
        self._set_infer_precision_hint()

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
        platform_list = self._devices_list()
        concurrency_device = {'CPU': 1, 'GPU': 1, 'HDDL': 100, 'MYRIAD': 4}
        if hasattr(self, 'exec_network') and self.exec_network is not None:
            if hasattr(self.exec_network, 'get_metric'):
                num_requests = self.exec_network.get_metric('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
            else:
                num_requests = self.exec_network.get_property('OPTIMAL_NUMBER_OF_INFER_REQUESTS')
            return num_requests
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
                    "'num_requests' option will be ignored.")
        elif 'num_requests' in self.config and self.config['num_requests'] != 'AUTO':
            num_per_device_requests = get_or_parse_value(self.config['num_request'], casting_type=int)
        else:
            num_per_device_requests = self.auto_num_requests(return_list=True)
        if len(num_per_device_requests) == 1:
            num_per_device_requests = [num_per_device_requests[0]] * len(device_list)
        if num_devices != len(num_per_device_requests):
            raise ConfigError('num requests for all {} should be specified'.format(num_devices))
        self._num_requests = sum(num_per_device_requests) * 2
        self._async_mode = True
        if log:
            self._log_versions()
            print_info('Async mode activated')

    def _set_device_config(self, device_config):
        if not isinstance(device_config, dict):
            raise ConfigError('device configuration should be a dict-like')
        if all(not isinstance(value, dict) for value in device_config.values()):
            ov_set_config(self.ie_core, dict(device_config), device=self.device)
        else:
            for key, value in device_config.items():
                if isinstance(value, dict):
                    if key in self._devices_list():
                        if key not in self.ie_core.available_devices:
                            warning('{} device is unknown. Config loading may lead to error.'.format(key))
                        ov_set_config(self.ie_core, dict(value), device=key)
                    else:
                        warning(
                            f'Configuration for {key} will be skipped as device is not listed in evaluation device')
                else:
                    warning(f'Option {key}: {value} will be skipped because device to which it should be '
                                  f'applied is not specified or option is not a dict-like')

    def _set_infer_precision_hint(self):
        precision_hint = self.config.get('_inference_precision_hint')
        if precision_hint is None:
            return
        supported_props = self.ie_core.get_property(self._device, 'SUPPORTED_PROPERTIES')
        if 'INFERENCE_PRECISION_HINT' not in supported_props:
            warning(f'inference precision hint is not supported for device {self._device}, option will be ingnored')
            return
        if precision_hint.upper() not in PRECISION_STR_TO_TYPE:
            warning(f'Unknown precision {precision_hint} for inference precision hint')
        precision_type = PRECISION_STR_TO_TYPE.get(precision_hint.upper(), precision_hint)
        self.ie_core.set_property(self._device, {'INFERENCE_PRECISION_HINT': precision_type})
        current_precision = self.ie_core.get_property(self._device, 'INFERENCE_PRECISION_HINT')
        print_info(f'Inference precision: {current_precision}')

    def _log_versions(self):
        versions = self.ie_core.get_versions(self._device)
        print_info("Loaded {} plugin version:".format(self._device))
        for device_name, device_version in versions.items():
            print_info("    {device_name} - {descr}: {maj}.{min}.{num}".format(
                device_name=device_name, descr=device_version.description, maj=device_version.major,
                min=device_version.minor, num=device_version.build_number))

    def _create_network(self, input_shapes=None):
        model_path = Path(self._model)
        compiled_model = model_path.suffix == '.blob'
        self.out_tensor_name_to_node = {}
        if compiled_model:
            self.network = None
            with open(str(self._model), 'rb') as f: #pylint:disable=unspecified-encoding
                self.exec_network = self.ie_core.import_model(io.BytesIO(f.read()), self._device)
            self.original_outputs = self.exec_network.outputs
            model_batch = self._get_model_batch_size()
            self._batch = model_batch if model_batch is not None else 1
            for out in self.original_outputs:
                if not out.names:
                    continue
                for name in out.names:
                    self.out_tensor_name_to_node[name] = out.get_node().friendly_name
            return
        if self._weights is None and self._model.suffix not in ['.onnx', '.pb', '.tflite']:
            self._weights = model_path.parent / (model_path.name.split(model_path.suffix)[0] + '.bin')
        self.network = self.read_network(self._model, self._weights)
        self.original_outputs = self.network.outputs
        for out in self.original_outputs:
            if not out.names:
                continue
            for name in out.names:
                self.out_tensor_name_to_node[name] = out.get_node().friendly_name
        model_batch = self._get_model_batch_size()
        model_batch = 1 if model_batch is None else model_batch
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
        self._batch = self.config.get('batch', model_batch)
        self._set_batch_size(self._batch)
        affinity_map_path = self.config.get('affinity_map')
        if affinity_map_path and self._is_hetero():
            self._set_affinity(affinity_map_path)
        elif affinity_map_path:
            warning('affinity_map config is applicable only for HETERO device')

    def _set_batch_size(self, batch_size):
        model_batch_size = self._get_model_batch_size()
        model_batch_size = 1 if model_batch_size is None else model_batch_size
        if batch_size is None:
            batch_size = model_batch_size
        if batch_size == model_batch_size:
            self._batch = batch_size
            return
        input_shapes = {}
        for input_node in self.network.inputs:
            layer_name = input_node.get_node().friendly_name
            if layer_name in self.const_inputs:
                input_shapes[layer_name] = parse_partial_shape(input_node.get_node().partial_shape)
            else:
                layer_shape = list(parse_partial_shape(input_node.get_node().partial_shape))
                layout = self._process_layout(self.inputs[layer_name].layout, layer_name)
                batch_pos = layout.find('N')
                if batch_pos != -1:
                    layer_shape[batch_pos] = batch_size
                input_shapes[layer_name] = layer_shape
        self._reshape_input(input_shapes, batch_size == -1)
        self._batch = batch_size

    def _process_layout(self, ov_layout, layer_name):
        if '...' in str(ov_layout) or ov_layout is None:
            ov_layout = self.get_layout_from_config(layer_name)
        else:
            ov_layout = str(ov_layout).replace('[', '').replace(']', '').replace(',', '')
        return ov_layout

    def _get_model_batch_size(self):
        input_nodes = self.network.inputs if self.network else self.exec_network.inputs
        input_info = input_nodes[0]
        layout = (
            self._process_layout(input_info.get_node().layout, input_info.get_node().friendly_name)
            or ''
        )
        input_shape = parse_partial_shape(input_info.partial_shape)
        if not layout and len(input_shape) == len(self.default_layout):
            layout = self.default_layout
        batch_pos = layout.find('N')
        if batch_pos != -1:
            return parse_partial_shape(input_info.partial_shape)[batch_pos]
        return None

    def load_network(self, network=None, log=False, preprocessing=None):
        if hasattr(self, 'exec_network'):
            del self.exec_network
        if hasattr(self, 'infer_request'):
            del self.infer_request
            self.infer_request = None
        if network is None:
            self._create_network()
        else:
            self.network = network
        if self.network is not None:
            self.dyn_input_layers, self._partial_shapes = self.get_dynamic_inputs(self.network)
        self.input_to_tensor_name = self.get_input_tensor_name_mapping(
            self.network if self.network is not None else self.exec_network)
        network_inputs = self.network.inputs if self.network is not None else self.exec_network.inputs
        self.input_to_index = {inp.get_node().friendly_name: idx for idx, inp in enumerate(network_inputs)}
        if not self._postpone_input_configuration:
            self._set_precision()
            self._set_input_shape()
            self.dyn_input_layers, self._partial_shapes = self.get_dynamic_inputs(self.network)
            if log:
                self.print_input_output_info(self.network if self.network is not None else self.exec_network)
            if preprocessing:
                self._set_preprocess(preprocessing)
                self.dyn_input_layers, self._partial_shapes = self.get_dynamic_inputs(self.network)
            model_batch = self._get_model_batch_size()
            model_batch = 1 if model_batch is None else model_batch
            self._batch = self.config.get('batch', model_batch)
            self._set_batch_size(self._batch)
            self.try_to_set_default_layout()
            if self.network and (not self.dyn_input_layers or self.is_dynamic or self.disable_resize_to_input):
                self.exec_network = self.ie_core.compile_model(self.network, self._device)
                self.infer_request = self.exec_network.create_infer_request()

    def update_input_configuration(self, input_config):
        self.config['inputs'] = input_config
        self._set_precision()
        self._set_input_shape()
        self.try_to_set_default_layout()
        self._set_batch_size(self.config.get('batch'))
        self.dyn_input_layers, self._partial_shapes = self.get_dynamic_inputs(self.network)
        self.print_input_output_info(self.network if self.network is not None else self.exec_network)
        if self.preprocessor:
            self._set_preprocess(self.preprocessor)
        if self.network:
            self.exec_network = self.ie_core.compile_model(self.network, self._device)
            self.infer_request = self.exec_network.create_infer_request()

    @staticmethod
    def get_dynamic_inputs(network):
        def is_dynamic(data_info):
            if hasattr(data_info, 'is_dynamic'):
                return data_info.is_dynamic
            return -1 in data_info.shape or not data_info.shape

        inputs_with_undefined_shapes = []
        partial_shapes = {}
        if network is None:
            return inputs_with_undefined_shapes, partial_shapes
        for input_info in network.inputs:
            input_node = input_info.get_node()
            input_shape = input_node.get_partial_shape()
            if is_dynamic(input_shape):
                inputs_with_undefined_shapes.append(input_node.friendly_name)
                partial_shapes[input_node.friendly_name] = input_shape
        return inputs_with_undefined_shapes, partial_shapes

    @staticmethod
    def get_input_tensor_name_mapping(network):
        inputs_mapping = {}
        for idx, input_node in enumerate(network.inputs):
            tensor_names = list(input_node.get_names())
            if not tensor_names:
                inputs_mapping[input_node.get_node().friendly_name] = idx
            else:
                inputs_mapping[input_node.get_node().friendly_name] = tensor_names[0]
        return inputs_mapping

    @property
    def nodel_input_mapping(self):
        input_mapping = {}
        for idx, input_node in enumerate(self.network.inputs):
            node_name = input_node.get_node().friendly_name
            input_mapping[idx] = input_node.get_node().friendly_name
            input_mapping[node_name] = node_name
            for tensor_name in list(input_node.get_names()):
                input_mapping[tensor_name] = node_name
        return input_mapping

    @property
    def dyn_batch_only(self):
        if not self.dyn_input_layers:
            return True
        for input_name in self.dyn_input_layers:
            partial_shape = self._partial_shapes[input_name]
            num_undef = 0
            for i in partial_shape:
                if i == -1:
                    num_undef += 1
                if num_undef > 1:
                    return False
            layout = self.inputs[input_name].layout
            if '...' in str(layout) or layout is None:
                layout = self.get_layout_from_config(input_name)
            else:
                layout = str(layout).replace('[', '').replace(']', '').replace(',', '')
            if not layout:
                return False
            for dim, layout_dim in zip(partial_shape, layout):
                if dim == -1 and layout_dim != 'N':
                    return False
        return True

    def get_layout_from_config(self, input_name):
        for input_config in self.config.get('inputs', []):
            if input_config.get('name', '') != input_name:
                continue
            return input_config.get('layout', '')
        return ''

    @property
    def layout_mapping(self):
        def prepare_layout_string(layout):
            layout = str(layout)
            return layout.replace('[', '').replace(']', '').replace(',', '')
        inputs = self.network.inputs if self.network is not None else self.exec_network.inputs
        layouts = {}
        for input_node in inputs:
            layouts[input_node.get_node().friendly_name] = prepare_layout_string(input_node.get_node().layout)
        return layouts

    def load_ir(self, xml_path, bin_path, log=False):
        self._model = xml_path
        self._weights = bin_path
        self.async_mode = True
        self.load_network(log=log)
        self.try_to_set_default_layout()

    def read_network(self, model, weights):
        if weights is not None:
            network = self.ie_core.read_model(model=str(model), weights=str(weights))
        else:
            network = self.ie_core.read_model(model=str(model))
        self.input_to_tensor_name = self.get_input_tensor_name_mapping(network)
        self.input_to_index = {inp.get_node().friendly_name: idx for idx, inp in enumerate(network.inputs)}
        return network

    def inputs_info_for_meta(self, inputs=None):
        if inputs:
            return {layer_name: np.shape(data) for layer_name, data in inputs.items()}
        return {
            layer_name: parse_partial_shape(layer.get_partial_shape()) for layer_name, layer in self.inputs.items()
            if layer_name not in self.const_inputs + self.image_info_inputs}

    @property
    def lstm_inputs(self):
        return self._lstm_inputs

    def initialize_undefined_shapes(self, input_data, template_shapes=None):
        if self.dynamic_shapes_policy in ['default', 'dynamic']:
            try:
                if template_shapes:
                    input_shapes = {layer_name: template_shapes.get(layer_name, data.shape)
                                    for layer_name, data in input_data[0].items()}
                    self._reshape_input(input_shapes)
                    self.load_network(self.network)
                    self.is_dynamic = True
                if not hasattr(self, 'exec_network') or self.exec_network is None:
                    self.is_dynamic = True
                    self.load_network(self.network)
                self.exec_network.infer_new_request({
                    self.input_to_tensor_name[k]: data for k, data in input_data[0].items()})
                return
            except RuntimeError as e:
                if self.dynamic_shapes_policy == 'dynamic':
                    raise e
                self.is_dynamic = False
        self._reshape_input({layer_name: data.shape for layer_name, data in input_data[0].items()})

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
            self.load_network(self.network)

    def fit_to_input(self, data, layer_name, layout, precision, template=None):
        if precision is None:
            precision = format_map[self.inputs[layer_name].element_type.get_type_name()]
        if layer_name in self.dyn_input_layers:
            layer_rang = len(parse_partial_shape(self._partial_shapes[layer_name]))
            input_template = template.get(layer_name) if template else template
            data, l_template = self._data_to_blob_dyn(layer_rang, data, layout, input_template)
            layer_shape = data.shape
            if l_template is not None:
                template[layer_name] = l_template
        else:
            layer_shape = tuple(self.inputs[layer_name].shape)
            precision = format_map[self.inputs[layer_name].element_type.get_type_name()]
            data = self._data_to_blob(layer_shape, data, layout)
        if precision:
            data = data.astype(precision)
        if layer_name in self.dyn_input_layers:
            self._do_reshape = not self.is_dynamic
            return data
        data_shape = np.shape(data)
        if data_shape != layer_shape:
            if self.allow_reshape_input:
                self._do_reshape = True
                return data
        return self._align_data_shape(data, layer_name, layout)

    @staticmethod
    def _data_to_blob_dyn(layer_rang, data, layout, template=None):
        data_shape = np.shape(data)
        if layer_rang - len(data_shape) == 1:
            data = np.expand_dims(data, 0)
            data_shape = np.shape(data)
            if template is not None:
                template = [1].extend(template)
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
        if layout and len(layout) == len(data_shape):
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
                if np.size(np.squeeze(np.zeros(layer_shape))) == np.size(np.squeeze(np.zeros(data_shape))):
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
            if layout is not None:
                return np.transpose(data[0], layout) if len(layout) == 3 else np.transpose(data, layout)[0]
            return data[0]
        if len(layer_shape) == 1:
            return np.resize(data, layer_shape)
        if (len(data_shape) == 3) and (len(layer_shape) == 2) and (data_shape[0] == 1) and (
                data_shape[1] == 1) and self.allow_reshape_input:
            return data[0]
        if layout is not None and len(layer_shape) == len(layout):
            return np.transpose(data, layout)
        if (len(layer_shape) == 1 and len(data_shape) > 1 and
            len(np.squeeze(np.zeros(layer_shape))) == len(np.squeeze(np.zeros(data_shape)))):
            return np.resize(data, layer_shape)
        return np.array(data)

    def _set_precision(self):
        config_inputs = self.config.get('inputs', [])
        has_precisions = ['precision' in inp for inp in config_inputs]
        if has_precisions and self.network:
            preprocessor = PrePostProcessor(self.network)
            for input_config in config_inputs:
                if 'precision' in input_config:
                    name = input_config['name']
                    element_type =  PRECISION_STR_TO_TYPE[input_config['precision'].upper()]
                    preprocessor.input(self.input_to_index[name]).tensor().set_element_type(element_type)
            self.network = preprocessor.build()

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
        orig_input_shapes = {input_name: parse_partial_shape(input_info.partial_shape)
                             for input_name, input_info in self.inputs.items()}
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
            layer_shape = parse_partial_shape(self.inputs[lstm_var].partial_shape)
            if infer_outputs:
                output_layer = postprocess_output_name(output_layer, infer_outputs)
            input_data = infer_outputs[output_layer].reshape(layer_shape) if infer_outputs else np.zeros(
                layer_shape, dtype=format_map[self.inputs[lstm_var].element_type.get_type_name()]
            )
            feed_dict[lstm_var] = input_data
        return feed_dict

    @staticmethod
    def print_input_output_info(network, prefix=None):
        if prefix:
            print_info('{} - Input info:'.format(prefix))
        else:
            print_info('Input info:')
        network_inputs = network.inputs
        network_outputs = network.outputs
        for input_info in network_inputs:
            input_node = input_info.get_node()
            print_info('\tNode name: {}'.format(input_node.friendly_name))
            print_info('\tTensor names: {}'.format(', '.join(input_info.get_names())))
            print_info('\tprecision: {}'.format(input_node.element_type.get_type_name()))
            print_info('\tshape: {}\n'.format(parse_partial_shape(input_node.get_partial_shape())))
        print_info('Output info')
        for output_info in network_outputs:
            out_node = output_info.get_node()
            print_info('\tNode name: {}'.format(out_node.friendly_name))
            print_info('\tTensor names: {}'.format(', '.join(output_info.get_names())))
            precision = out_node.get_output_element_type(0).get_type_name()
            print_info('\tprecision: {}'.format(precision))
            shape = parse_partial_shape(out_node.get_output_partial_shape(0))
            print_info('\tshape: {}\n'.format(shape))

    def _set_preprocess(self, preprocess):
        if preprocess.ie_processor is None:
            return
        if self.network is not None:
            self.disable_resize_to_input = False
            preprocess_steps = preprocess.ie_preprocess_steps
            if not preprocess_steps:
                return
            preprocessor = PrePostProcessor(self.network)
            for input_name in self.inputs:
                if input_name in self.const_inputs + self.image_info_inputs:
                    continue
                input_id = self.input_to_index[input_name]
                for (name, value) in preprocess_steps:
                    if name == 'resize_algorithm':
                        preprocessor.input(input_id).tensor().set_spatial_dynamic_shape()
                        preprocessor.input(input_id).preprocess().resize(value)
                        self.need_dyn_resolving = False
                    if name == 'convert_color_format':
                        src, dst = value
                        preprocessor.input(input_id).tensor().set_color_format(src)
                        preprocessor.input(input_id).preprocess().convert_color(dst)
                    if name == 'mean_variant':
                        mean, scale = value
                        if mean is not None:
                            preprocessor.input(input_id).preprocess().mean(mean)
                        if scale is not None:
                            preprocessor.input(input_id).preprocess().scale(scale)
            self.network = preprocessor.build()
            self._preprocess_steps = preprocess_steps
        self.disable_resize_to_input = preprocess.ie_processor.has_resize()

    def get_model_file_type(self):
        if hasattr(self, '_model'):
            return self._model.suffix
        return None

    def get_infer_queue(self, log=True):
        if self.config.get('num_requests', 'AUTO') == 'AUTO':
            num_requests = self.auto_num_requests()
        else:
            num_requests = self.num_requests or 0
        queue = AsyncInferQueue(self.exec_network, num_requests)
        if log:
            print_info('Prepared async infer queue with {} requests'.format(len(queue)))
        else:
            debug('Prepared async infer queue with {} requests'.format(len(queue)))
        return queue

    def prepare_data_for_request(self, inputs, batch_meta, batch_id, batch_input_ids,
                                 batch_annotation, batch_identifiers):
        infer_inputs = inputs[0]
        feed_dict = {self.input_to_tensor_name[name]: data for name, data in infer_inputs.items()}
        if batch_meta is not None:
            self._fill_meta(batch_meta, None if not self.dyn_input_layers else infer_inputs)
        context = (batch_id, batch_input_ids, batch_annotation, batch_identifiers, batch_meta)
        return feed_dict, context

    @staticmethod
    def get_result_from_request(request, return_raw=False):
        preprocessed_results = [{out.get_node().friendly_name: data for out, data in request.results.items()}]
        if return_raw:
            return preprocessed_results, [request.results]
        return preprocessed_results

    def input_shape(self, input_name):
        return parse_partial_shape(self.inputs[input_name].get_partial_shape())

    def release(self):
        if 'network' in self.__dict__:
            del self.network
        if 'infer_request' in self.__dict__:
            del self.infer_request
        if 'exec_network' in self.__dict__:
            del self.exec_network
        if 'ie_core' in self.__dict__:
            del self.ie_core
