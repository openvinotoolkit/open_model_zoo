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

from pathlib import Path
import sys

import math
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

from models import Model

class DeblurringModel(Model):
    def __init__(self, ie, model_path, input_image_shape):
        super().__init__(ie, model_path)
        self.calculate_new_shape(input_image_shape)
        self.input_blob_name = self.prepare_inputs()
        self.output_blob_name = self.prepare_outputs()

    def calculate_new_shape(self, shape):
        h, w, _ = shape
        block_size = 32
        new_height = math.ceil(h / block_size) * block_size
        new_width = math.ceil(w / block_size) * block_size

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
        if len(output_size) == 4:
            self.out_channels = output_size[1]
            self.out_height = output_size[2]
            self.out_width = output_size[3]
        else:
            raise Exception("Unexpected output blob shape {}. Only 4D output blob is supported".format(output_size))

        return output_blob_name

    @staticmethod
    def _normalize(image, mean=127.5, std=127.5):
        image = image - mean
        image = image / std
        return image

    @staticmethod
    def _padding(image, pad_params):
        return np.pad(image, **pad_params)

    def preprocess(self, inputs):
        image = inputs
        self.mean = 127.5
        self.std = 127.5
        normalized_image = self._normalize(image, self.mean, self.std)
        # right bottom padding to resize input image to input_layer shape
        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, self.h - image.shape[0]), (0, self.w - image.shape[1]), (0, 0))
                      }

        if image.shape[0] != self.h or image.shape[1] != self.w:
            resized_image = self._padding(normalized_image, pad_params)
        else:
            resized_image = normalized_image
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = np.expand_dims(resized_image, 0)

        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        dict_inputs = {self.input_blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        prediction = outputs[self.output_blob_name].squeeze()
        input_image_height = meta['original_shape'][0]
        input_image_width = meta['original_shape'][1]

        prediction = prediction.transpose((1, 2, 0))
        prediction *= self.std
        prediction += self.mean
        prediction = prediction.astype(np.uint8)

        return prediction[:min(input_image_height, self.h), :min(input_image_width, self.w), :]
