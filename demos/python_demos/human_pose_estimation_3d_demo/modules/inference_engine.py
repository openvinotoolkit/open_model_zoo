"""
 Copyright (c) 2019 Intel Corporation
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

import os

import numpy as np

from openvino.inference_engine import IECore


class InferenceEngine:
    def __init__(self, net_model_xml_path, device, stride):
        self.device = device
        self.stride = stride

        self.ie = IECore()

        self.net = self.ie.read_network(net_model_xml_path, os.path.splitext(net_model_xml_path)[0] + '.bin')
        required_input_key = {'data'}
        assert required_input_key == set(self.net.input_info), \
            'Demo supports only topologies with the following input key: {}'.format(', '.join(required_input_key))
        required_output_keys = {'features', 'heatmaps', 'pafs'}
        assert required_output_keys.issubset(self.net.outputs.keys()), \
            'Demo supports only topologies with the following output keys: {}'.format(', '.join(required_output_keys))

        self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=device)

    def infer(self, img):
        img = img[0:img.shape[0] - (img.shape[0] % self.stride),
                  0:img.shape[1] - (img.shape[1] % self.stride)]
        input_layer = next(iter(self.net.input_info))
        n, c, h, w = self.net.input_info[input_layer].input_data.shape
        if h != img.shape[0] or w != img.shape[1]:
            self.net.reshape({input_layer: (n, c, img.shape[0], img.shape[1])})
            self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=self.device)
        img = np.transpose(img, (2, 0, 1))[None, ]

        inference_result = self.exec_net.infer(inputs={'data': img})

        inference_result = (inference_result['features'][0],
                            inference_result['heatmaps'][0], inference_result['pafs'][0])
        return inference_result
