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
import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

class ImageInpainting(object):
    def __init__(self, ie, model_path, parts, max_brush_width, max_length, max_vertex, device='CPU'):
        model = IENetwork(model=model_path, weights=os.path.splitext(model_path)[0] + '.bin')

        assert len(model.inputs) == 2, "Expected 2 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blobs"

        self._input_layer_names = sorted(model.inputs)
        self._output_layer_name = next(iter(model.outputs))

        self._ie = ie
        self._exec_model = self._ie.load_network(model, device)
        self.infer_time = -1

        _, channels, input_height, input_width = model.inputs[self._input_layer_names[0]].shape
        assert channels == 3, "Expected 3-channel input"

        _, channels, mask_height, mask_width = model.inputs[self._input_layer_names[1]].shape
        assert channels == 1, "Expected 1-channel input"

        assert mask_height == input_height and mask_width == input_width, "Mask size expected to be equal to image size"
        self.input_height = input_height
        self.input_width = input_width

        self.parts = parts
        self.max_brush_width = max_brush_width
        self.max_length = max_length
        self.max_vertex = max_vertex

    @staticmethod
    def _free_form_mask(mask, max_vertex, max_length, max_brush_width, h, w, max_angle=360):
        num_strokes = np.random.randint(max_vertex)
        start_y = np.random.randint(h)
        start_x = np.random.randint(w)
        brush_width = 0
        for i in range(num_strokes):
            angle = np.random.random() * np.deg2rad(max_angle)
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(max_length + 1)
            brush_width = np.random.randint(10, max_brush_width + 1) // 2 * 2
            next_y = start_y + length * np.cos(angle)
            next_x = start_x + length * np.sin(angle)

            next_y = np.clip(next_y, 0, h - 1).astype(np.int)
            next_x = np.clip(next_x, 0, w - 1).astype(np.int)
            cv2.line(mask, (start_y, start_x), (next_y, next_x), 1, brush_width)
            cv2.circle(mask, (start_y, start_x), brush_width // 2, 1)

            start_y, start_x = next_y, next_x
        return mask

    def preprocess(self, image):
        image = cv2.resize(image, (self.input_width, self.input_height))
        mask = np.zeros((self.input_height, self.input_width, 1), dtype=np.float32)

        for _ in range(self.parts):
            mask = self._free_form_mask(mask, self.max_vertex, self.max_length, self.max_brush_width,
                                         self.input_height, self.input_width)

        image = image * (1 - mask) + 255 * mask
        return image, mask

    def infer(self, image, mask):
        t0 = cv2.getTickCount()
        output = self._exec_model.infer(inputs={self._input_layer_names[0]: image, self._input_layer_names[1]: mask})
        self.infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        return output[self._output_layer_name]

    def process(self, image):
        masked_image, mask = self.preprocess(image)
        image = np.transpose(masked_image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        output = self.infer(image, mask)

        output = np.transpose(output, (0, 2, 3, 1)).astype(np.uint8)
        output[0] = cv2.cvtColor(output[0], cv2.COLOR_RGB2BGR)
        masked_image = masked_image.astype(np.uint8)
        return masked_image, output[0]
