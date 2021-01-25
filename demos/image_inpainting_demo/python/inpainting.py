"""
 Copyright (c) 2019-2020 Intel Corporation
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


class ImageInpainting:
    def __init__(self, ie, model_path, device='CPU'):
        model = ie.read_network(model_path, model_path.with_suffix('.bin'))

        assert len(model.input_info) == 2, "Expected 2 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blobs"

        self._input_layer_names = sorted(model.input_info)
        self._output_layer_name = next(iter(model.outputs))

        self._ie = ie
        self._exec_model = self._ie.load_network(model, device)
        self.infer_time = -1

        _, channels, input_height, input_width = model.input_info[self._input_layer_names[0]].input_data.shape
        assert channels == 3, "Expected 3-channel input"

        _, channels, mask_height, mask_width = model.input_info[self._input_layer_names[1]].input_data.shape
        assert channels == 1, "Expected 1-channel input"

        assert mask_height == input_height and mask_width == input_width, "Mask size expected to be equal to image size"
        self.input_height = input_height
        self.input_width = input_width


    def infer(self, image, mask):
        t0 = cv2.getTickCount()
        output = self._exec_model.infer(inputs={self._input_layer_names[0]: image, self._input_layer_names[1]: mask})
        self.infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        return output[self._output_layer_name]


    def process(self, src_image, mask):
        image = np.transpose(src_image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        output = self.infer(image, mask)

        output = np.transpose(output, (0, 2, 3, 1)).astype(np.uint8)
        return output[0]
