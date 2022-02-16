"""
 Copyright (c) 2020-2022 Intel Corporation
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
    """Class for inference of models in the Inference Engine format"""
    def __init__(self, core, model_path, labels_file, conf=.6, device='CPU', ext_path=''):
        self.confidence = conf
        self.load_model(core, model_path, device, ext_path)
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
        res = self.infer_request.infer(inputs={self.input_tensor_name: self._preprocess(img)})
        return next(iter(res.values()))

    def get_detections(self, input):
        raise NotImplementedError

    def get_input_shape(self):
        """Returns an input shape of the wrapped IE model"""
        return self.model.inputs[0].shape

    def get_allowed_inputs_len(self):
        return (1, )

    def get_allowed_outputs_len(self):
        return (1, )

    def load_model(self, core, model_path, device, cpu_extension=''):
        """Loads a model in the Inference Engine format"""
        # Plugin initialization for specified device and load extensions library if specified
        if cpu_extension and 'CPU' in device:
            core.add_extension(cpu_extension, 'CPU')
        # Read IR
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
        self.infer_request = compiled_model.create_infer_request()
