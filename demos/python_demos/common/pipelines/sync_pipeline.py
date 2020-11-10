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


class SyncPipeline:
    def __init__(self, ie, model, device='CPU'):
        self.device = device
        self.model = model

        self.exec_net = ie.load_network(network=self.model.net, device_name=self.device)

    def submit_data(self, inputs):
        inputs, meta = self.model.preprocess(inputs)
        outputs = self.exec_net.infer(inputs=inputs)
        outputs = self.model.postprocess(outputs, meta)
        return outputs, meta
