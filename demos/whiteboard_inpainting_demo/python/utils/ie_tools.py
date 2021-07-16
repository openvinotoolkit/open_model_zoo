"""
 Copyright (c) 2020 Intel Corporation
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

import cv2
import numpy as np
import os


class IEModel:
    """Class for inference of models in the Inference Engine format"""
    def __init__(self, ie, model_path, labels_file, conf=.6, device='CPU', ext_path=''):
        self.confidence = conf
        self.load_ie_model(ie, model_path, device, ext_path)
        with open(labels_file, 'r') as f:
            self.labels = f.readlines()
        self.labels = {num: name.replace('\n', '') for num, name in enumerate(self.labels)}
        self.classes_to_hide = self.set_classes_to_hide()
        self.labels_to_hide = [num for num, name in self.labels.items() if name in self.classes_to_hide]

    @staticmethod
    def set_classes_to_hide():
        return ('person', )

    def _preprocess(self, img):
        _, _, h, w = self.get_input_shape()
        img = np.expand_dims(cv2.resize(img, (w, h)).transpose(2, 0, 1), axis=0)
        return img

    def forward(self, img):
        """Performs forward pass of the wrapped IE model"""
        res = self.net.infer(inputs={self.input_key: self._preprocess(img)})
        return res[self.output_key]

    def get_detections(self, input):
        raise NotImplementedError

    def get_input_shape(self):
        """Returns an input shape of the wrapped IE model"""
        return self.inputs_info[self.input_key].input_data.shape

    def get_allowed_inputs_len(self):
        return (1, )

    def get_allowed_outputs_len(self):
        return (1, )

    def load_ie_model(self, ie, model_xml, device, cpu_extension=''):
        """Loads a model in the Inference Engine format"""
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Plugin initialization for specified device and load extensions library if specified
        if cpu_extension and 'CPU' in device:
            ie.add_extension(cpu_extension, 'CPU')
        # Read IR
        net = ie.read_network(model=model_xml, weights=model_bin)

        assert len(net.input_info) in self.get_allowed_inputs_len(), \
            "Supports topologies with only {} inputs, but got {}" \
            .format(self.get_allowed_inputs_len(), len(net.input_info))
        assert len(net.outputs) in self.get_allowed_outputs_len(), \
            "Supports topologies with only {} outputs, but got {}" \
            .format(self.get_allowed_outputs_len(), len(net.outputs))

        input_blob = next(iter(net.input_info))
        out_blob = next(iter(net.outputs))
        net.batch_size = 1

        # Loading model to the plugin
        self.net = ie.load_network(network=net, device_name=device)
        self.inputs_info = net.input_info
        self.input_key = input_blob
        self.output_key = out_blob
