"""
 Copyright (c) 2018-2022 Intel Corporation

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
from openvino.runtime import AsyncInferQueue


class Module:
    def __init__(self, core, model, model_type):
        self.core = core
        self.model_type = model_type
        log.info('Reading {} model {}'.format(model_type, model))
        self.model = core.read_model(model, model.with_suffix('.bin'))
        self.model_path = model
        self.active_requests = 0
        self.clear()

    def deploy(self, device, plugin_config, max_requests=1):
        self.max_requests = max_requests
        compiled_model = self.core.compile_model(self.model, device, config=plugin_config)
        self.infer_queue = AsyncInferQueue(compiled_model, self.max_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(self.model_type, self.model_path, device))

    def completion_callback(self, infer_request, id):
        self.outputs[id] = next(iter(infer_request.results.values()))

    def enqueue(self, input):
        self.clear()

        if self.max_requests <= self.active_requests:
            log.warning('Processing request rejected - too many requests')
            return False

        self.infer_queue.start_async(input, self.active_requests)
        self.active_requests += 1
        return True

    def get_outputs(self):
        if self.active_requests <= 0:
            return
        self.infer_queue.wait_all()
        self.active_requests = 0
        return [v for _, v in sorted(self.outputs.items())]

    def clear(self):
        self.outputs = {}

    def infer(self, inputs):
        self.clear()
        self.start_async(*inputs)
        return self.postprocess()
