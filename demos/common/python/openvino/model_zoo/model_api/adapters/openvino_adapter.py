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

from .model_adapter import ModelAdapter, Metadata


class OpenvinoAdapter(ModelAdapter):
    """
    Class that allows working with Inference Engine model, its input and output blobs
    """

    def __init__(self, core, model_path, plugin_config, device, max_num_requests=1):
        self.core = core
        self.model_path = model_path
        self.plugin_config = plugin_config
        self.device = device
        self.max_num_requests = max_num_requests
        self.net = core.read_network(model_path)

    def load_model(self):
        self.exec_net = self.core.load_network(self.net, self.device,
            self.plugin_config, self.max_num_requests)
        if self.max_num_requests == 0:
            # ExecutableNetwork doesn't allow creation of additional InferRequests. Reload ExecutableNetwork
            # +1 to use it as a buffer of the pipeline
            self.exec_net = self.core.load_network(self.net, self.device,
                self.plugin_config, len(self.exec_net.requests) + 1)

        log.info('The model {} is loaded to {}'.format(self.model_path, self.device))
        self.empty_requests = deque(self.exec_net.requests)

    def get_input_layers(self):
        inputs = {}
        for name, layer in self.net.input_info.items():
            inputs[name] = Metadata(layer.input_data.shape, layer.input_data.precision)
        return inputs

    def get_output_layers(self):
        outputs = {}
        for name, layer in self.net.outputs.items():
            outputs[name] = Metadata(layer.shape, layer.precision)
        return outputs

    def reshape_model(self, new_shape):
        self.net.reshape(new_shape)

    def infer_sync(self, dict_data):
        return self.exec_net.infer(dict_data)

    def infer_async(self, dict_data, callback_fn, callback_data):

        def get_raw_result(request):
            self.empty_requests.append(request)
            return {key: blob.buffer for key, blob in request.output_blobs.items()}

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
