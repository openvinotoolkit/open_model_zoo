"""
 Copyright (c) 2021-2022 Intel Corporation
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

from .model import WrapperError
from .image_model import ImageModel


class Deblurring(ImageModel):
    __model__ = 'Deblurring'

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self._check_io_number(1, 1)
        self.block_size = 32
        self.output_blob_name = self._get_outputs()

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        return parameters

    def reshape(self, base_shape):
        h, w, _ = base_shape
        new_height = math.ceil(h / self.block_size) * self.block_size
        new_width = math.ceil(w / self.block_size) * self.block_size
        self.h, self.w = new_height, new_width
        self.logger.debug("\tReshape model from {} to {}".format(
            [self.n, self.c, h, w], [self.n, self.c, self.h, self.w]))
        super().reshape({self.image_blob_name: [self.n, self.c, self.h, self.w]})

    def _get_outputs(self):
        output_blob_name = next(iter(self.outputs))
        output_size = self.outputs[output_blob_name].shape
        if len(output_size) != 4:
            raise WrapperError(self.__model__, "Unexpected output blob shape {}. Only 4D output blob is supported".format(output_size))

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
            self.logger.warning("\tChosen model size doesn't match image size. The image is resized")
            resized_image = cv2.resize(image, (self.w, self.h))

        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = np.expand_dims(resized_image, 0)
        dict_inputs = {self.image_blob_name: resized_image}
        meta = {'original_shape': image.shape[1::-1]}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        dsize = meta['original_shape']
        prediction = outputs[self.output_blob_name].squeeze()
        prediction = prediction.transpose((1, 2, 0))
        if self.h - self.block_size < dsize[1] <= self.h and self.w - self.block_size < dsize[0] <= self.w:
            prediction = prediction[:dsize[1], :dsize[0], :]
        else:
            prediction = cv2.resize(prediction, dsize)
        prediction *= 255
        return prediction.astype(np.uint8)
