"""
 Copyright (c) 2021 Intel Corporation
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
import math
import numpy as np

from .model import Model

class Deblurring(Model):
    def __init__(self, ie, model_path, input_image_shape):
        super().__init__(ie, model_path)
        self.block_size = 32
        self.reshape(input_image_shape)
        self.input_blob_name = self.prepare_inputs()
        self.output_blob_name = self.prepare_outputs()

    def reshape(self, base_shape):
        h, w, _ = base_shape
        new_height = math.ceil(h / self.block_size) * self.block_size
        new_width = math.ceil(w / self.block_size) * self.block_size

        input_layer = next(iter(self.net.input_info))
        input_shape = self.net.input_info[input_layer].input_data.shape
        input_shape[2:] = (new_height, new_width)
        self.net.reshape({input_layer: input_shape})

    def prepare_inputs(self):
        input_num = len(self.net.input_info)
        if input_num != 1:
            raise RuntimeError("Demo supports topologies only with 1 input")

        input_blob_name = next(iter(self.net.input_info))
        input_blob = self.net.input_info[input_blob_name]
        input_blob.precision = "FP32"

        input_size = input_blob.input_data.shape
        if len(input_size) == 4 and input_size[1] == 3:
            self.n, self.c, self.h, self.w = input_size
        else:
            raise RuntimeError("3-channel 4-dimensional model's input is expected")

        return input_blob_name

    def prepare_outputs(self):
        output_num = len(self.net.outputs)
        if output_num != 1:
            raise RuntimeError("Demo supports topologies only with 1 output")

        output_blob_name = next(iter(self.net.outputs))
        output_blob = self.net.outputs[output_blob_name]
        output_blob.precision = "FP32"

        output_size = output_blob.shape
        if len(output_size) != 4:
            raise Exception("Unexpected output blob shape {}. Only 4D output blob is supported".format(output_size))

        return output_blob_name

    def preprocess(self, inputs):
        image = inputs

        if self.h - self.block_size < image.shape[0] <= self.h and self.w - self.block_size < image.shape[1] <= self.w:
            pad_params = {'mode': 'constant',
                          'constant_values': 0,
                          'pad_width': ((0, self.h - image.shape[0]), (0, self.w - image.shape[1]), (0, 0))
                          }
            resized_image = np.pad(image, **pad_params)
        else:
            self.logger.warn("Chosen model size doesn't match image size. The image is resized")
            resized_image = cv2.resize(image, (self.w, self.h))

        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = np.expand_dims(resized_image, 0)

        dict_inputs = {self.input_blob_name: resized_image}
        return dict_inputs, image.shape[1::-1]

    def postprocess(self, outputs, dsize):
        prediction = outputs[self.output_blob_name].squeeze()
        prediction = prediction.transpose((1, 2, 0))
        if self.h - self.block_size < dsize[1] <= self.h and self.w - self.block_size < dsize[0] <= self.w:
            prediction = prediction[:dsize[1], :dsize[0], :]
        else:
            prediction = cv2.resize(prediction, dsize)
        prediction *= 255
        return prediction.astype(np.uint8)
