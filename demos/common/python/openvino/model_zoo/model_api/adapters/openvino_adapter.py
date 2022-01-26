"""
 Copyright (c) 2021 Intel Corporation

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
    from openvino.runtime import AsyncInferQueue, Core, PartialShape, get_version
    openvino_absent = False
except ImportError:
    openvino_absent = True

from .model_adapter import ModelAdapter, Metadata
from ..pipelines import parse_devices


def create_core():
    if openvino_absent:
        raise ImportError('The OpenVINO package is not installed')

    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    return Core()


class OpenvinoAdapter(ModelAdapter):
    """
    Works with OpenVINO model
    """

    def __init__(self, core, model_path, weights_path=None, device='CPU', plugin_config=None, max_num_requests=0):
        self.core = core
        self.model_path = model_path
        self.device = device
        self.plugin_config = plugin_config
        self.max_num_requests = max_num_requests

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

        log.info('The model {} is loaded to {}'.format("from buffer" if self.model_from_buffer else self.model_path, self.device))
        if self.max_num_requests == 0:
            self.max_num_requests = self.get_optimal_number_of_requests()
        self.async_queue = AsyncInferQueue(self.compiled_model, self.max_num_requests)
        self.log_runtime_settings()

    def get_optimal_number_of_requests(self):
        metrics = self.compiled_model.get_metric('SUPPORTED_METRICS')
        key = 'OPTIMAL_NUMBER_OF_INFER_REQUESTS'
        if key in metrics:
            return self.compiled_model.get_metric(key) + 1
        return 1

    def log_runtime_settings(self):
        devices = set(parse_devices(self.device))
        if 'AUTO' not in devices:
            for device in devices:
                try:
                    nstreams = self.compiled_model.get_config(device + '_THROUGHPUT_STREAMS')
                    log.info('\tDevice: {}'.format(device))
                    log.info('\t\tNumber of streams: {}'.format(nstreams))
                    if device == 'CPU':
                        nthreads = self.compiled_model.get_config('CPU_THREADS_NUM')
                        log.info('\t\tNumber of threads: {}'.format(nthreads if int(nthreads) else 'AUTO'))
                except RuntimeError:
                    pass
        log.info('\tNumber of network infer requests: {}'.format(len(self.async_queue)))

    def get_input_layers(self):
        inputs = {}
        for input in self.model.inputs:
            inputs[input.get_any_name()] = Metadata(input.get_names(), list(input.shape), input.get_element_type().get_type_name())
        inputs = self._get_meta_from_ngraph(inputs)
        return inputs

    def get_output_layers(self):
        outputs = {}
        for output in self.model.outputs:
            output_shape = output.partial_shape.get_min_shape() if self.model.is_dynamic() else output.shape
            outputs[output.get_any_name()] = Metadata(output.get_names(), list(output_shape), output.get_element_type().get_type_name())
        outputs = self._get_meta_from_ngraph(outputs)
        return outputs

    def reshape_model(self, new_shape):
        new_shape = {k: PartialShape(v) for k, v in new_shape.items()}
        self.model.reshape(new_shape)

    def get_raw_result(self, request):
        raw_result = {key: request.get_tensor(key).data[:] for key in self.get_output_layers().keys()}
        return raw_result

    def infer_sync(self, dict_data):
        request = self.compiled_model.create_infer_request()
        request.infer(dict_data)
        return self.get_raw_result(request)

    def infer_async(self, dict_data, callback_data):
        self.async_queue.start_async(dict_data, (self.get_raw_result, callback_data))

    def is_ready(self):
        return self.async_queue.is_ready()

    def await_all(self):
        self.async_queue.wait_all()

    def await_any(self):
        self.async_queue.is_ready()

    def _get_meta_from_ngraph(self, layers_info):
        for node in self.model.get_ordered_ops():
            layer_name = node.get_friendly_name()
            if layer_name not in layers_info.keys():
                continue
            layers_info[layer_name].meta = node.get_attributes()
            layers_info[layer_name].type = node.get_type_name()
        return layers_info
