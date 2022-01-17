"""
 Copyright (c) 2018-2022 Intel Corporation

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

from utils import cut_rois, resize_input
from ie_module import Module


class LandmarksDetector(Module):
    POINTS_NUMBER = 5

    def __init__(self, core, model):
        super(LandmarksDetector, self).__init__(core, model, 'Landmarks Detection')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        output_shape = self.model.outputs[0].shape
        if not np.array_equal([1, self.POINTS_NUMBER * 2, 1, 1], output_shape):
            raise RuntimeError("The model expects output shape {}, got {}".format(
                [1, self.POINTS_NUMBER * 2, 1, 1], output_shape))

    def preprocess(self, frame, rois):
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape, self.nchw_layout) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(LandmarksDetector, self).enqueue({self.input_tensor_name: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def postprocess(self):
        results = [out.reshape((-1, 2)).astype(np.float64) for out in self.get_outputs()]
        return results
