"""
 Copyright (c) 2018 Intel Corporation

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
import os.path as osp

from openvino.inference_engine import IECore

class InferenceContext:
    def __init__(self, devices, cpu_ext, gpu_ext, perf_count):
        self.ie_core = IECore()
        self.gpu_ext = gpu_ext
        self.perf_count = perf_count

        if cpu_ext and 'CPU' in devices:
            log.info("Using CPU extensions library '%s'" % (cpu_ext))
            self.ie_core.add_extension(cpu_ext, "CPU")

    def check_model_support(self, net, device):
        if device == "CPU":
            supported_layers = self.ie_core.query_network(net, device)
            not_supported_layers = [l for l in net.layers.keys() \
                                    if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("The following layers are not supported " \
                    "by the plugin for the specified device {}:\n {}" \
                    .format(plugin.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions " \
                    "library path in the command line parameters using " \
                    "the '-l' parameter")
                raise NotImplementedError(
                    "Some layers are not supported on the device")

    def deploy_model(self, model, device, max_requests=1):
        self.check_model_support(model, device)

        config = {
            "PERF_COUNT": "YES" if self.perf_count else "NO",
        }

        if device == 'GPU' and self.gpu_ext:
            config['CONFIG_FILE'] = self.gpu_ext

        return self.ie_core.load_network(model, device, config=config, num_requests=max_requests)



class Module(object):
    def __init__(self, model):
        self.model = model
        self.device_model = None

        self.max_requests = 0
        self.active_requests = 0

        self.clear()

    def deploy(self, device, context, queue_size=1):
        self.context = context
        self.max_requests = queue_size
        self.device_model = context.deploy_model(
            self.model, device, self.max_requests)
        self.model = None

    def enqueue(self, input):
        self.clear()

        if self.max_requests <= self.active_requests:
            log.warning("Processing request rejected - too many requests")
            return False

        self.device_model.start_async(self.active_requests, input)
        self.active_requests += 1
        return True

    def wait(self):
        if self.active_requests <= 0:
            return

        self.perf_stats = [None, ] * self.active_requests
        self.outputs = [None, ] * self.active_requests
        for i in range(self.active_requests):
            self.device_model.requests[i].wait()
            self.outputs[i] = self.device_model.requests[i].output_blobs
            self.perf_stats[i] = self.device_model.requests[i].get_perf_counts()

        self.active_requests = 0

    def get_outputs(self):
        self.wait()
        return self.outputs

    def get_performance_stats(self):
        return self.perf_stats

    def clear(self):
        self.perf_stats = []
        self.outputs = []
