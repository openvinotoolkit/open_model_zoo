"""
 Copyright (c) 2020-2024 Intel Corporation
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


class IEModel:
    """Class for inference of models in the OpenVINO Runtime format"""
    def __init__(self, core, model_path, labels_file, conf=.6, device='CPU'):
        self.confidence = conf
        self.load_model(core, model_path, device)
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
        """Performs forward pass of the wrapped model"""
        input_data = {self.input_tensor_name: self._preprocess(img)}
        return self.infer_request.infer(input_data)[self.output_tensor]

    def get_detections(self, input):
        raise NotImplementedError

    def get_input_shape(self):
        """Returns an input shape of the wrapped model"""
        return self.model.inputs[0].shape

    def get_allowed_inputs_len(self):
        return (1, )

    def get_allowed_outputs_len(self):
        return (1, )

    def load_model(self, core, model_path, device):
        """Loads a model in the OpenVINO Runtime format"""
        self.model = core.read_model(model_path)

        if len(self.model.inputs) not in self.get_allowed_inputs_len():
            raise RuntimeError("Supports topologies with only {} inputs, but got {}"
                .format(self.get_allowed_inputs_len(), len(self.model.inputs)))
        if len(self.model.outputs) not in self.get_allowed_outputs_len():
            raise RuntimeError("Supports topologies with only {} outputs, but got {}"
                .format(self.get_allowed_outputs_len(), len(self.model.outputs)))

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        # Loading model to the plugin
        compiled_model = core.compile_model(self.model, device)
        self.output_tensor = compiled_model.outputs[0]
        self.infer_request = compiled_model.create_infer_request()
