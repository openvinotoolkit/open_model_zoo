"""
 Copyright (c) 2021-2023 Intel Corporation

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

import logging as log
from pathlib import Path

try:
    from openvino.runtime import AsyncInferQueue, Core, PartialShape, layout_helpers, get_version, Dimension
    openvino_absent = False
except ImportError:
    openvino_absent = True

from .model_adapter import ModelAdapter, Metadata
from .utils import Layout
from ..pipelines import parse_devices


def create_core():
    if openvino_absent:
        raise ImportError('The OpenVINO package is not installed')

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    return Core()


class OpenvinoAdapter(ModelAdapter):
    '''
    Works with OpenVINO model
    '''

    def __init__(self, core, model_path, weights_path=None, model_parameters = {}, device='CPU', plugin_config=None, max_num_requests=0):
        self.core = core
        self.model_path = model_path
        self.device = device
        self.plugin_config = plugin_config
        self.max_num_requests = max_num_requests
        self.model_parameters = model_parameters
        self.model_parameters['input_layouts'] = Layout.parse_layouts(self.model_parameters.get('input_layouts', None))

        if isinstance(model_path, (str, Path)):
            if Path(model_path).suffix == ".onnx" and weights_path:
                log.warning('For model in ONNX format should set only "model_path" parameter.'
                            'The "weights_path" will be omitted')

        self.model_from_buffer = isinstance(model_path, bytes) and isinstance(weights_path, bytes)
        log.info('Reading model {}'.format('from buffer' if self.model_from_buffer else model_path))
        weights = weights_path if self.model_from_buffer else ''
        self.model = core.read_model(model_path, weights)

    def load_model(self):
        self.compiled_model = self.core.compile_model(self.model, self.device, self.plugin_config)
        self.async_queue = AsyncInferQueue(self.compiled_model, self.max_num_requests)
        if self.max_num_requests == 0:
            # +1 to use it as a buffer of the pipeline
            self.async_queue = AsyncInferQueue(self.compiled_model, len(self.async_queue) + 1)

        log.info('The model {} is loaded to {}'.format("from buffer" if self.model_from_buffer else self.model_path, self.device))
        self.log_runtime_settings()

    def log_runtime_settings(self):
        devices = set(parse_devices(self.device))
        if 'AUTO' not in devices:
            for device in devices:
                try:
                    nstreams = self.compiled_model.get_property(device + '_THROUGHPUT_STREAMS')
                    log.info('\tDevice: {}'.format(device))
                    log.info('\t\tNumber of streams: {}'.format(nstreams))
                    if device == 'CPU':
                        nthreads = self.compiled_model.get_property('CPU_THREADS_NUM')
                        log.info('\t\tNumber of threads: {}'.format(nthreads if int(nthreads) else 'AUTO'))
                except RuntimeError:
                    pass
        log.info('\tNumber of model infer requests: {}'.format(len(self.async_queue)))

    def get_input_layers(self):
        inputs = {}
        for input in self.model.inputs:
            input_shape = get_input_shape(input)
            input_layout = self.get_layout_for_input(input, input_shape)
            inputs[input.get_any_name()] = Metadata(input.get_names(), input_shape, input_layout, input.get_element_type().get_type_name())
        inputs = self._get_meta_from_ngraph(inputs)
        return inputs

    def get_layout_for_input(self, input, shape=None) -> str:
        input_layout = ''
        if self.model_parameters['input_layouts']:
            input_layout = Layout.from_user_layouts(input.get_names(), self.model_parameters['input_layouts'])
        if not input_layout:
            if not layout_helpers.get_layout(input).empty:
                input_layout = Layout.from_openvino(input)
            else:
                input_layout = Layout.from_shape(shape if shape is not None else input.shape)
        return input_layout

    def get_output_layers(self):
        outputs = {}
        for output in self.model.outputs:
            output_shape = output.partial_shape.get_min_shape() if self.model.is_dynamic() else output.shape
            outputs[output.get_any_name()] = Metadata(output.get_names(), list(output_shape), precision=output.get_element_type().get_type_name())
        outputs = self._get_meta_from_ngraph(outputs)
        return outputs

    def reshape_model(self, new_shape):
        new_shape = {name: PartialShape(
            [Dimension(dim) if not isinstance(dim, tuple) else Dimension(dim[0], dim[1])
            for dim in shape]) for name, shape in new_shape.items()}
        self.model.reshape(new_shape)

    def get_raw_result(self, request):
        return {key: request.get_tensor(key).data for key in self.get_output_layers()}

    def copy_raw_result(self, request):
        return {key: request.get_tensor(key).data.copy() for key in self.get_output_layers()}

    def infer_sync(self, dict_data):
        self.infer_request = self.async_queue[self.async_queue.get_idle_request_id()]
        self.infer_request.infer(dict_data)
        return self.get_raw_result(self.infer_request)

    def infer_async(self, dict_data, callback_data) -> None:
        self.async_queue.start_async(dict_data, (self.copy_raw_result, callback_data))

    def set_callback(self, callback_fn):
        self.async_queue.set_callback(callback_fn)

    def is_ready(self) -> bool:
        return self.async_queue.is_ready()

    def await_all(self) -> None:
        self.async_queue.wait_all()

    def await_any(self) -> None:
        self.async_queue.get_idle_request_id()

    def _get_meta_from_ngraph(self, layers_info):
        for node in self.model.get_ordered_ops():
            layer_name = node.get_friendly_name()
            if layer_name not in layers_info.keys():
                continue
            layers_info[layer_name].meta = node.get_attributes()
            layers_info[layer_name].type = node.get_type_name()
        return layers_info

    def operations_by_type(self, operation_type):
        layers_info = {}
        for node in self.model.get_ordered_ops():
            if node.get_type_name() == operation_type:
                layer_name = node.get_friendly_name()
                layers_info[layer_name] = Metadata(type=node.get_type_name(), meta=node.get_attributes())
        return layers_info


def get_input_shape(input_tensor):
    def string_to_tuple(string, casting_type=int):
        processed = string.replace(' ', '').replace('(', '').replace(')', '').split(',')
        processed = filter(lambda x: x, processed)
        return tuple(map(casting_type, processed)) if casting_type else tuple(processed)
    if not input_tensor.partial_shape.is_dynamic:
        return list(input_tensor.shape)
    ps = str(input_tensor.partial_shape)
    if ps[0] == '[' and ps[-1] == ']':
        ps = ps[1:-1]
    preprocessed = ps.replace('{', '(').replace('}', ')').replace('?', '-1')
    preprocessed = preprocessed.replace('(', '').replace(')', '')
    if '..' in preprocessed:
        shape_list = []
        for dim in preprocessed.split(','):
            if '..' in dim:
                shape_list.append(string_to_tuple(dim.replace('..', ',')))
            else:
                shape_list.append(int(dim))
        return shape_list
    return string_to_tuple(preprocessed)
