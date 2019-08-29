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

import sys
import os

import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IECore # pylint: disable=import-error,E0611
import cv2 as cv


class IEModel:
    """Class for inference of models in the Inference Engine format"""
    def __init__(self, exec_net, inputs_info, input_key, output_key):
        self.net = exec_net
        self.inputs_info = inputs_info
        self.input_key = input_key
        self.output_key = output_key
        self.reqs_ids = []

    def _preprocess(self, img):
        _, _, h, w = self.get_input_shape().shape
        img = np.expand_dims(cv.resize(img, (w, h)).transpose(2, 0, 1), axis=0)
        return img

    def forward(self, img):
        """Performs forward pass of the wrapped IE model"""
        res = self.net.infer(inputs={self.input_key: self._preprocess(img)})
        return np.copy(res[self.output_key])

    def forward_async(self, img):
        id = len(self.reqs_ids)
        self.net.start_async(request_id=id,
                             inputs={self.input_key: self._preprocess(img)})
        self.reqs_ids.append(id)

    def grab_all_async(self):
        outputs = []
        for id in self.reqs_ids:
            self.net.requests[id].wait(-1)
            res = self.net.requests[id].outputs[self.output_key]
            outputs.append(np.copy(res))
        self.reqs_ids = []
        return outputs

    def get_input_shape(self):
        """Returns an input shape of the wrapped IE model"""
        return self.inputs_info[self.input_key]


def load_ie_model(model_xml, device, plugin_dir, cpu_extension='', num_reqs=1):
    """Loads a model in the Inference Engine format"""
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if cpu_extension and 'CPU' in device:
        ie.add_extension(cpu_extension, 'CPU')
    # Read IR
    log.info("Loading network files:\n\t%s\n\t%s", model_xml, model_bin)
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if not_supported_layers:
            log.error("Following layers are not supported by the plugin for specified device %s:\n %s",
                      device, ', '.join(not_supported_layers))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1 or len(net.inputs.keys()) == 2, \
        "Supports topologies with only 1 or 2 inputs"
    assert len(net.outputs) == 1 or len(net.outputs) == 5, \
        "Supports topologies with only 1 or 5 outputs"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device, num_requests=num_reqs)
    model = IEModel(exec_net, net.inputs, input_blob, out_blob)
    return model
