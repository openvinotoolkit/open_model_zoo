"""
 Copyright (c) 2019-2022 Intel Corporation
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

import numpy as np
import cv2


class IEModel:
    """Class for inference of models in the Inference Engine format"""
    def __init__(self, core, model_path, device, num_reqs=1, cpu_extension=''):
        self.load_model(core, model_path, device, num_reqs, cpu_extension)
        self.reqs_ids = []

    def _preprocess(self, img):
        _, _, h, w = self.get_input_shape()
        img = np.expand_dims(cv2.resize(img, (w, h)).transpose(2, 0, 1), axis=0)
        return img

    def forward(self, img):
        """Performs forward pass of the wrapped IE model"""
        self.infer_requests[0].infer({self.input_tensor_name: self._preprocess(img)})
        return self.infer_requests[0].get_tensor(self.output_tensor_name).data[:]

    def forward_async(self, img):
        id = len(self.reqs_ids)
        self.infer_requests[id].start_async({self.input_tensor_name: self._preprocess(img)})
        self.reqs_ids.append(id)

    def grab_all_async(self):
        outputs = []
        for id in self.reqs_ids:
            self.infer_requests[id].wait()
            res = self.infer_requests[id].get_tensor(self.output_tensor_name).data[:]
            outputs.append(res)
        self.reqs_ids = []
        return outputs

    def get_allowed_inputs_len(self):
        return (1, 2)

    def get_allowed_outputs_len(self):
        return (1, 2, 3, 4, 5)

    def get_input_shape(self):
        """Returns an input shape of the wrapped IE model"""
        return self.model.inputs[0].shape

    def load_model(self, core, model_xml, device, num_reqs=1, cpu_extension=''):
        """Loads a model in the Inference Engine format"""
        # Plugin initialization for specified device and load extensions library if specified
        if cpu_extension and 'CPU' in device:
            core.add_extension(cpu_extension, 'CPU')
        # Read IR
        self.model = core.read_model(model_xml)

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
        self.infer_requests = [compiled_model.create_infer_request() for _ in range(num_reqs)]
