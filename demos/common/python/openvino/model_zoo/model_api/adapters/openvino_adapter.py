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
from collections import deque
from pathlib import Path

try:
    from openvino.inference_engine import IECore, get_version
    import ngraph
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
    return IECore()


class OpenvinoAdapter(ModelAdapter):
    """
    Class that allows working with Inference Engine model, its input and output blobs
    """

    def __init__(self, ie, model_path, weights_path=None, device='CPU', plugin_config=None, max_num_requests=1):
        self.ie = ie
        self.model_path = model_path
        self.device = device
        self.plugin_config = plugin_config
        self.max_num_requests = max_num_requests

        if isinstance(model_path, (str, Path)):
            model_path_suffix = Path(model_path).suffix
            if model_path_suffix == ".onnx":
                if weights_path:
                    log.warning('For model in ONNX format should set only "model_path" parameter.'
                                'The "weights_path" will be omitted')
                    weights_path = None
            elif model_path_suffix == ".xml":
                weights_path_suffix = Path(weights_path).suffix if weights_path else None
                if weights_path_suffix and weights_path_suffix != ".bin":
                    raise ValueError(f"Unsupported weights file extension: {weights_path_suffix}")
            else:
                raise ValueError(f"Unsupported model file extension: {model_path_suffix}")

        self.model_from_buffer = isinstance(model_path, bytes) and isinstance(weights_path, bytes)
        log.info('Reading model {}'.format('from buffer' if self.model_from_buffer else model_path))
        self.net = ie.read_network(model_path, weights_path, self.model_from_buffer)

    def load_model(self):
        self.exec_net = self.ie.load_network(self.net, self.device,
            self.plugin_config, self.max_num_requests)
        if self.max_num_requests == 0:
            # ExecutableNetwork doesn't allow creation of additional InferRequests. Reload ExecutableNetwork
            # +1 to use it as a buffer of the pipeline
            self.exec_net = self.ie.load_network(self.net, self.device,
                self.plugin_config, len(self.exec_net.requests) + 1)

        log.info('The model {} is loaded to {}'.format("from buffer" if self.model_from_buffer else self.model_path, self.device))
        self.empty_requests = deque(self.exec_net.requests)
        self.log_runtime_settings()

    def log_runtime_settings(self):
        devices = set(parse_devices(self.device))
        if 'AUTO' not in devices:
            for device in devices:
                try:
                    nstreams = self.exec_net.get_config(device + '_THROUGHPUT_STREAMS')
                    log.info('\tDevice: {}'.format(device))
                    log.info('\t\tNumber of streams: {}'.format(nstreams))
                    if device == 'CPU':
                        nthreads = self.exec_net.get_config('CPU_THREADS_NUM')
                        log.info('\t\tNumber of threads: {}'.format(nthreads if int(nthreads) else 'AUTO'))
                except RuntimeError:
                    pass
        log.info('\tNumber of network infer requests: {}'.format(len(self.exec_net.requests)))

    def get_input_layers(self):
        inputs = {}
        for name, layer in self.net.input_info.items():
            inputs[name] = Metadata(layer.input_data.shape, layer.input_data.precision)
        inputs = self._get_meta_from_ngraph(inputs)
        return inputs

    def get_output_layers(self):
        outputs = {}
        for name, layer in self.net.outputs.items():
            outputs[name] = Metadata(layer.shape, layer.precision)
        outputs = self._get_meta_from_ngraph(outputs)
        return outputs

    def reshape_model(self, new_shape):
        self.net.reshape(new_shape)

    def infer_sync(self, dict_data):
        return self.exec_net.infer(dict_data)

    def infer_async(self, dict_data, callback_fn, callback_data):

        def get_raw_result(request):
            raw_result = {key: blob.buffer for key, blob in request.output_blobs.items()}
            self.empty_requests.append(request)
            return raw_result

        request = self.empty_requests.popleft()
        request.set_completion_callback(py_callback=callback_fn,
                                        py_data=(get_raw_result, request, callback_data))
        request.async_infer(dict_data)

    def is_ready(self):
        return len(self.empty_requests) != 0

    def await_all(self):
        for request in self.exec_net.requests:
            request.wait()

    def await_any(self):
        self.exec_net.wait(num_requests=1)

    def _get_meta_from_ngraph(self, layers_info):
        ng_func = ngraph.function_from_cnn(self.net)
        for node in ng_func.get_ordered_ops():
            layer_name = node.get_friendly_name()
            if layer_name not in layers_info.keys():
                continue
            layers_info[layer_name].meta = node._get_attributes()
            layers_info[layer_name].type = node.get_type_name()
        return layers_info
