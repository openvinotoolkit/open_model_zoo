"""
 Copyright (c) 2018 Intel Corporation

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

    class Result:
        def __init__(self, outputs):
            self.points = outputs

            p = lambda i: self[i]
            self.left_eye = p(0)
            self.right_eye = p(1)
            self.nose_tip = p(2)
            self.left_lip_corner = p(3)
            self.right_lip_corner = p(4)
        def __getitem__(self, idx):
            return self.points[idx]

        def get_array(self):
            return np.array(self.points, dtype=np.float64)

    def __init__(self, model):
        super(LandmarksDetector, self).__init__(model)

        assert len(model.input_info) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.input_info))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.input_info[self.input_blob].input_data.shape

        assert np.array_equal([1, self.POINTS_NUMBER * 2, 1, 1],
                              model.outputs[self.output_blob].shape), \
            "Expected model output shape %s, but got %s" % \
            ([1, self.POINTS_NUMBER * 2, 1, 1],
             model.outputs[self.output_blob].shape)

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(LandmarksDetector, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_landmarks(self):
        outputs = self.get_outputs()
        results = [LandmarksDetector.Result(out[self.output_blob].buffer.reshape((-1, 2))) \
                      for out in outputs]
        return results
