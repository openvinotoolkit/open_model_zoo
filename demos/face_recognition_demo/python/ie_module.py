"""
 Copyright (c) 2018-2021 Intel Corporation

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

import logging


class Module:
    def __init__(self, ie, model):
        self.ie = ie
        self.model = ie.read_network(model, model.with_suffix('.bin'))
        self.active_requests = 0
        self.clear()

    def deploy(self, device, plugin_config, max_requests=1):
        self.max_requests = max_requests
        self.exec_net = self.ie.load_network(self.model, device, config=plugin_config, num_requests=max_requests)

    def enqueue(self, input):
        self.clear()

        if self.max_requests <= self.active_requests:
            logging.warning('Processing request rejected - too many requests')
            return False

        self.exec_net.start_async(self.active_requests, input)
        self.active_requests += 1
        return True

    def wait(self):
        if self.active_requests <= 0:
            return

        self.perf_stats = [None, ] * self.active_requests
        self.outputs = [None, ] * self.active_requests
        for i in range(self.active_requests):
            self.exec_net.requests[i].wait()
            self.outputs[i] = self.exec_net.requests[i].output_blobs
            self.perf_stats[i] = self.exec_net.requests[i].get_perf_counts()

        self.active_requests = 0

    def get_outputs(self):
        self.wait()
        return self.outputs

    def get_performance_stats(self):
        return self.perf_stats

    def clear(self):
        self.perf_stats = []
        self.outputs = []

    def infer(self, inputs):
        self.clear()
        self.start_async(*inputs)
        return self.postprocess()
