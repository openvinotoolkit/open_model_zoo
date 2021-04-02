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
import numpy as np

class Model:
    def __init__(self, ie, model_path, reverse_input_channels=None, mean_values=None, scale_values=None):
        self.logger = logging.getLogger()
        self.logger.info('Reading network from IR...')
        self.is_onnx_format = model_path.suffix == '.onnx'
        if self.is_onnx_format:
            self.net = ie.read_network(model_path)
        else:
            self.net = ie.read_network(model_path, model_path.with_suffix('.bin'))
        self.set_batch_size(1)

        self.reverse_input_channels = reverse_input_channels
        self.mean_values = np.array(mean_values, dtype=np.float32) if mean_values else None
        self.scale_values = np.array(scale_values, dtype=np.float32) if scale_values else None

    def preprocess(self, inputs):
        meta = {}
        return inputs, meta

    def postprocess(self, outputs, meta):
        return outputs

    def set_batch_size(self, batch):
        shapes = {}
        for input_layer in self.net.input_info:
            new_shape = [batch] + self.net.input_info[input_layer].input_data.shape[1:]
            shapes.update({input_layer: new_shape})
        self.net.reshape(shapes)

    def scaleshift(self, inputs):
        if self.scale_values is None:
            return inputs - self.mean_values[:, None, None]
        return (inputs - self.mean_values[:, None, None]) / self.scale_values[:, None, None]
