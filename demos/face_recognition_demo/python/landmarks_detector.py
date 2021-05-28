"""
 Copyright (c) 2018-2021 Intel Corporation

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

    def __init__(self, ie, model):
        super(LandmarksDetector, self).__init__(ie, model)

        assert len(self.model.input_info) == 1, 'Expected 1 input blob'
        assert len(self.model.outputs) == 1, 'Expected 1 output blob'
        self.input_blob = next(iter(self.model.input_info))
        self.output_blob = next(iter(self.model.outputs))
        self.input_shape = self.model.input_info[self.input_blob].input_data.shape
        output_shape = self.model.outputs[self.output_blob].shape

        assert np.array_equal([1, self.POINTS_NUMBER * 2, 1, 1], output_shape), \
            'Expected model output shape {}, got {}'.format([1, self.POINTS_NUMBER * 2, 1, 1], output_shape)

    def preprocess(self, frame, rois):
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(LandmarksDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def postprocess(self):
        outputs = self.get_outputs()
        results = [out[self.output_blob].buffer.reshape((-1, 2)).astype(np.float64) for out in outputs]
        return results
