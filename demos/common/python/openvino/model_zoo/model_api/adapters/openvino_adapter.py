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

import threading
import logging as log
from collections import deque

from openvino.inference_engine import IECore, get_version

from .model_adapter import ModelAdapter


class OpenvinoAdapter(ModelAdapter):
    """
    Class that allows working with Inference Engine model, its input and output blobs
    """

    def __init__(self, model_path, plugin_config, device, max_num_requests=1):
        log.info('OpenVINO Inference Engine')
        log.info('\tbuild: {}'.format(get_version()))
        self.core = IECore()

        self.load_model(model_path, plugin_config, device, max_num_requests)
        log.info('The model {} is loaded to {}'.format(model_path, device))

        self.empty_requests = deque(self.exec_net.requests)
        self.event = threading.Event()
        self.callback_exceptions = []

    def load_model(self, model_path, plugin_config, device, max_num_requests):
        self.net = self.core.read_network(model_path)
        self.exec_net = self.core.load_network(network=self.net, device_name=device,
            config=plugin_config, num_requests=max_num_requests)
        if max_num_requests == 0:
            # ExecutableNetwork doesn't allow creation of additional InferRequests. Reload ExecutableNetwork
            # +1 to use it as a buffer of the pipeline
            self.exec_net = self.core.load_network(network=self.net, device_name=device,
                config=plugin_config, num_requests=len(self.exec_net.requests) + 1)

    def get_input_layers(self):
        return list(self.net.input_info.keys())

    def get_output_layers(self):
        return list(self.net.outputs.keys())

    def get_input_layer_shape(self, input_layer_name):
        return self.net.input_info[input_layer_name].input_data.shape

    def get_output_layer_shape(self, output_layer_name):
        return self.net.outputs[output_layer_name].shape

    def get_input_layer_precision(self, input_layer_name):
        return self.net.input_info[input_layer_name].precision

    def get_output_layer_precision(self, output_layer_name):
        return self.net.outputs[output_layer_name].precision

    def create_infer_request(self, input_layer_name, data):
        return {input_layer_name: data}

    def infer(self, infer_request):
        return self.exec_net.infer(infer_request)

    def reshape_model(self, new_shape):
        self.net.reshape({self.image_blob_name: new_shape})

    def get_model(self):
        return self.exec_net

    def inference_completion_callback(self, status, callback_args):
        try:
            request, completed_results, (id, meta, preprocessing_meta, start_time) = callback_args
            if status != 0:
                raise RuntimeError('Infer Request has returned status code {}'.format(status))
            raw_outputs = {key: blob.buffer for key, blob in request.output_blobs.items()}
            completed_results[id] = (raw_outputs, meta, preprocessing_meta, start_time)
            self.empty_requests.append(request)
        except Exception as e:
            self.callback_exceptions.append(e)
        self.event.set()

    def async_infer(self, inputs, completed_results, callback_data):
        request = self.empty_requests.popleft()
        if len(self.empty_requests) == 0:
            self.event.clear()
        request.set_completion_callback(py_callback=self.inference_completion_callback,
                                        py_data=(request, completed_results, callback_data))
        request.async_infer(inputs)

    def is_ready(self):
        return len(self.empty_requests) != 0

    def await_all(self):
        for request in self.exec_net.requests:
            request.wait()

    def await_any(self):
        if len(self.empty_requests) == 0:
            self.event.wait()

    def check_exceptions(self):
        if self.callback_exceptions:
            raise self.callback_exceptions[0]
