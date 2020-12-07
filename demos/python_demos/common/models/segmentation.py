"""
 Copyright (c) 2020 Intel Corporation
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

from .model import Model
from .utils import resize_image

class SegmentationModel(Model):
    def __init__(self, ie, model_path):
        super().__init__(ie, model_path)
        self.out_height = 0
        self.out_width = 0
        self.out_channels = 0

        self.blob_name = self.prepare_inputs()
        self.n, self.c, self.h, self.w = self.net.input_info[self.blob_name].input_data.shape
        self.prepare_outputs()

    def prepare_inputs(self):
        input_shapes = self.net.input_info
        if len(input_shapes) != 1:
            raise RuntimeError("Demo supports topologies only with 1 input")
        blob_name = next(iter(self.net.input_info))
        blob = self.net.input_info[blob_name]

        in_size_vector = blob.input_data.shape
        if len(in_size_vector) != 4 or in_size_vector[1] != 3:
            print('ERROR IN PREPARE_INPUTS 2')
            raise RuntimeError("3-channel 4-dimensional model's input is expected")

        blob.layout = "NHWC"
        blob.precision = "U8"
        return blob_name

    def prepare_outputs(self):
        if len(self.net.outputs) != 1:
            raise RuntimeError("Demo supports topologies only with 1 output")

        out_blob_name = next(iter(self.net.outputs))
        blob = self.net.outputs[out_blob_name]
        blob.precision = "U8"
        out_size_vector = blob.shape

        if len(out_size_vector) == 3:
            self.out_channels = 0
            self.out_height = out_size_vector[1]
            self.out_width = out_size_vector[2]

        elif len(out_size_vector) == 4:
            self.out_channels = out_size_vector[1]
            self.out_height = out_size_vector[2]
            self.out_width = out_size_vector[3]

        else:
            raise Exception("Unexpected output blob shape {}. Only 4D and 3D output blobs are supported".format(out_size_vector))

    def preprocess(self, inputs):
        image = inputs
        resized_image = resize_image(image, (self.w, self.h))
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))

        dict_inputs = {self.blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        orginal_image_shape = meta['original_shape']
        result = np.zeros(shape=(self.out_height, self.out_width, 3), dtype=np.uint8)

        for row_id in range(self.out_height):
            for col_id in range(self.out_width):
                class_id = 0
                if self.out_channels < 2: # assume the output is already ArgMax'ed
                    class_id = outputs[row_id * self.out_width + col_id]
                else:
                    max_prob = -1.0
                    for ch_id in range(self.out_channels):
                        prob = outputs[ch_id * self.out_height * self.out_width + row_id * self.out_width + col_id]
                        if prob > max_prob:
                            class_id = ch_id
                            max_prob = prob
                result[row_id, col_id] = class_id
        result = cv2.resize(result, orginal_image_shape, cv2.INTER_NEAREST)
        return result
