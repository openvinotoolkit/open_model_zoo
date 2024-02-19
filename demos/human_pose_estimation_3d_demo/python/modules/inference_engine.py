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

from openvino import Core, get_version, PartialShape


class InferenceEngine:
    def __init__(self, model_path, device, stride):
        self.device = device
        self.stride = stride

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        self.core = Core()

        log.info('Reading model {}'.format(model_path))
        self.model = self.core.read_model(model_path)

        required_output_keys = {'features', 'heatmaps', 'pafs'}
        for output_tensor_name in required_output_keys:
            try:
                self.model.output(output_tensor_name)
            except RuntimeError:
                raise RuntimeError("The demo supports only topologies with the following output keys: {}".format(
                    ', '.join(required_output_keys)))

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        compiled_model = self.core.compile_model(self.model, self.device)
        self.infer_request = compiled_model.create_infer_request()
        log.info('The model {} is loaded to {}'.format(model_path, self.device))

    def infer(self, img):
        img = img[0:img.shape[0] - (img.shape[0] % self.stride),
                  0:img.shape[1] - (img.shape[1] % self.stride)]
        n, c, h, w = self.model.inputs[0].shape
        if h != img.shape[0] or w != img.shape[1]:
            self.model.reshape({self.input_tensor_name: PartialShape([n, c, img.shape[0], img.shape[1]])})
            compiled_model = self.core.compile_model(self.model, self.device)
            self.infer_request = compiled_model.create_infer_request()
        img = np.transpose(img, (2, 0, 1))[None, ]

        self.infer_request.infer({self.input_tensor_name: img})
        inference_result = {name: self.infer_request.get_tensor(name).data[:] for name in {'features', 'heatmaps', 'pafs'}}

        inference_result = (inference_result['features'][0],
                            inference_result['heatmaps'][0], inference_result['pafs'][0])
        return inference_result
