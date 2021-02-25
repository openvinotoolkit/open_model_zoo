"""
 Copyright (C) 2021 Intel Corporation

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


class SyncPipeline:
    def __init__(self, ie, model=None, device='CPU', silent=False):
        if silent:
            self.logger = logging.getLogger('dummy')
            self.logger.setLevel(logging.CRITICAL)
        else:
            self.logger = logging.getLogger()

        self.ie = ie
        self.device = device
        if model:
            self.model = model
            self.logger.info('Loading network to {} plugin...'.format(device))
            self.exec_net = ie.load_network(network=self.model.net, device_name=device)

    def reload_model(self, model, device=None):
        self.model = model
        self.logger.info('Loading network to {} plugin...'.format(device if device else self.device))
        self.exec_net = self.ie.load_network(network=self.model.net, device_name=device if device else self.device)

    def infer(self, inputs):
        inputs, preprocessing_meta = self.model.preprocess(inputs)
        raw_outputs = self.exec_net.infer(inputs)
        result = self.model.postprocess(raw_outputs, preprocessing_meta)[0]
        return result
