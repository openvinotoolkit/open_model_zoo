"""
 Copyright (c) 2019-2024 Intel Corporation
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

import numpy as np
import cv2

from openvino import AsyncInferQueue


class IEModel:
    """Class for inference of models in the OpenVINO Runtime format"""
    def __init__(self, core, model_path, device, model_type, num_reqs=1):
        self.load_model(core, model_path, device, model_type, num_reqs)
        self.outputs = {}

    def _preprocess(self, img):
        _, _, h, w = self.get_input_shape()
        img = np.expand_dims(cv2.resize(img, (w, h)).transpose(2, 0, 1), axis=0)
        return img

    def completion_callback(self, infer_request, id):
        self.outputs[id] = infer_request.get_tensor(self.output_tensor_name).data[:]

    def forward(self, img):
        """Performs forward pass of the wrapped model"""
        self.forward_async(img, 0)
        self.infer_queue.wait_all()
        return self.outputs.pop(0)

    def forward_async(self, img, req_id):
        input_data = {self.input_tensor_name: self._preprocess(img)}
        self.infer_queue.start_async(input_data, req_id)

    def grab_all_async(self):
        self.infer_queue.wait_all()
        return [self.outputs.pop(i) for i in range(len(self.outputs))]

    def get_allowed_inputs_len(self):
        return (1, 2)

    def get_allowed_outputs_len(self):
        return (1, 2, 3, 4, 5)

    def get_input_shape(self):
        """Returns an input shape of the wrapped model"""
        return self.model.inputs[0].shape

    def load_model(self, core, model_path, device, model_type, num_reqs=1):
        """Loads a model in the OpenVINO Runtime format"""

        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)

        if len(self.model.inputs) not in self.get_allowed_inputs_len():
            raise RuntimeError("Supports topologies with only {} inputs, but got {}"
                .format(self.get_allowed_inputs_len(), len(self.model.inputs)))
        if len(self.model.outputs) not in self.get_allowed_outputs_len():
            raise RuntimeError("Supports topologies with only {} outputs, but got {}"
                .format(self.get_allowed_outputs_len(), len(self.model.outputs)))

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.output_tensor_name = self.model.outputs[0].get_any_name()
        # Loading model to the plugin
        compiled_model = core.compile_model(self.model, device)
        self.infer_queue = AsyncInferQueue(compiled_model, num_reqs)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(model_type, model_path, device))
