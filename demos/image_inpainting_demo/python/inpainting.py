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


class ImageInpainting:
    def __init__(self, core, model_path, device='CPU'):
        model = core.read_model(model_path, model_path.with_suffix('.bin'))

        if len(model.inputs) != 2:
            raise RuntimeError("The model expects 2 input layers")
        if len(model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.image_input_layer, self.mask_input_layer = sorted([node.get_any_name() for node in model.inputs])

        compiled_model = core.compile_model(model, device)
        self.infer_request = compiled_model.create_infer_request()

        _, channels, input_height, input_width = model.input(self.image_input_layer).shape
        if channels != 3:
            raise RuntimeError("The model expects 3 channels for {} input layer".format(self.image_input_layer))

        _, channels, mask_height, mask_width = model.input(self.mask_input_layer).shape
        if channels != 1:
            raise RuntimeError("The model expects 3 channels for {} input layer".format(self.mask_input_layer))

        if mask_height != input_height or mask_width != input_width:
            raise RuntimeError("Mask size is expected to be equal to image size")
        self.input_height = input_height
        self.input_width = input_width


    def infer(self, image, mask):
        output = self.infer_request.infer(inputs={self.image_input_layer: image, self.mask_input_layer: mask})
        return next(iter(output.values()))


    def process(self, src_image, mask):
        image = np.transpose(src_image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        output = self.infer(image, mask)

        output = np.transpose(output, (0, 2, 3, 1)).astype(np.uint8)
        return output[0]
