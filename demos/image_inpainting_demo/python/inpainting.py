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

import numpy as np


class ImageInpainting:
    def __init__(self, core, model_path, device='CPU'):
        model = core.read_model(model_path)

        if len(model.inputs) != 2:
            raise RuntimeError("The model expects 2 input layers")
        if len(model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.image_input_layer, self.mask_input_layer = sorted([node.get_any_name() for node in model.inputs])

        compiled_model = core.compile_model(model, device)
        self.output_tensor = compiled_model.outputs[0]
        self.infer_request = compiled_model.create_infer_request()

        self.nchw_layout = model.input(self.image_input_layer).shape[1] == 3
        if self.nchw_layout:
            _, _, input_height, input_width = model.input(self.image_input_layer).shape
            _, mask_channels, mask_height, mask_width = model.input(self.mask_input_layer).shape
        else:
            _, input_height, input_width, _ = model.input(self.image_input_layer).shape
            _, mask_height, mask_width, mask_channels = model.input(self.mask_input_layer).shape

        if mask_channels != 1:
            raise RuntimeError("The model expects 1 channel for {} input layer".format(self.mask_input_layer))

        if mask_height != input_height or mask_width != input_width:
            raise RuntimeError("Mask size is expected to be equal to image size")
        self.input_height = input_height
        self.input_width = input_width

    def infer(self, image, mask):
        input_data = {self.image_input_layer: image, self.mask_input_layer: mask}
        return self.infer_request.infer(input_data)[self.output_tensor]

    def process(self, image, mask):
        if self.nchw_layout:
            image = np.transpose(image, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        output = self.infer(image, mask)

        if self.nchw_layout:
            output = np.transpose(output, (0, 2, 3, 1))
        return output.astype(np.uint8)[0]
