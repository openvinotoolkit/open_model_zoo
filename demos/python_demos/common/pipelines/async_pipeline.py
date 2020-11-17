"""
 Copyright (C) 2020 Intel Corporation

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
from collections import deque
from time import perf_counter


class AsyncPipeline:
    def __init__(self, ie, model, logger=None, device='CPU', plugin_config={}, max_num_requests=1,
                 completed_requests=None, caught_exceptions=None):
        self.model = model
        self.logger = logger

        if self.logger:
            self.logger.info('Loading network to {} plugin...'.format(device))
        loading_time = perf_counter()
        self.exec_net = ie.load_network(network=self.model.net, device_name=device,
                                        config=plugin_config, num_requests=max_num_requests)
        loading_time = (perf_counter() - loading_time)
        if self.logger:
            self.logger.info('Loaded in {:.3f} seconds'.format(loading_time))

        self.empty_requests = deque(self.exec_net.requests)
        self.completed_request_results = completed_requests if completed_requests else {}
        self.callback_exceptions = caught_exceptions if caught_exceptions else {}
        self.event = threading.Event()

    def inference_completion_callback(self, status, callback_args):
        request, id, meta = callback_args
        try:
            if status != 0:
                raise RuntimeError('Infer Request has returned status code {}'.format(status))
            raw_outputs = {key: blob.buffer for key, blob in request.output_blobs.items()}
            self.completed_request_results[id] = (raw_outputs, meta)
            self.empty_requests.append(request)
        except Exception as e:
            self.callback_exceptions.append(e)
        self.event.set()

    def submit_data(self, inputs, id, meta):
        request = self.empty_requests.popleft()
        inputs, preprocessing_meta = self.model.preprocess(inputs)
        meta.update(preprocessing_meta)
        request.set_completion_callback(py_callback=self.inference_completion_callback,
                                        py_data=(request, id, meta))
        self.event.clear()
        request.async_infer(inputs=inputs)

    def get_raw_result(self, id):
        if id in self.completed_request_results:
            return self.completed_request_results.pop(id)
        return None

    def get_result(self, id):
        result = self.get_raw_result(id)
        if result:
            return self.model.postprocess(*result), result[1]
        return None

    def await_all(self):
        for request in self.exec_net.requests:
            request.wait()

    def await_any(self):
        if len(self.empty_requests) == 0:
            self.event.wait()
