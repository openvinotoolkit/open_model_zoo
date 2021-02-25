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

import logging


class Model:
    def __init__(self, ie, model_path):
        self.logger = logging.getLogger()
        self.logger.info('Reading network from IR...')
        self.net = ie.read_network(model_path, model_path.with_suffix('.bin'))
        self.set_batch_size(1)

    def preprocess(self, inputs):
        meta = {}
        return inputs, meta

    def postprocess(self, outputs, meta):
        return outputs

    def reshape(self, shapes):
        if isinstance(shapes, dict):
            self.net.reshape(shapes)
        elif isinstance(shapes, list) and len(self.net.input_info) == 1:
            self.net.reshape({next(iter(self.net.input_info)): shapes})
        else:
            raise NotImplementedError

    def set_batch_size(self, batch):
        shapes = {}
        for input_layer in self.net.input_info:
            new_shape = [batch] + self.net.input_info[input_layer].input_data.shape[1:]
            shapes.update({input_layer: new_shape})
        self.net.reshape(shapes)
