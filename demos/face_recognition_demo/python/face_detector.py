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
from numpy import clip
from ie_module import Module
from utils import resize_input

class FaceDetector(Module):
    class Result:
        OUTPUT_SIZE = 7

        def __init__(self, output):
            self.image_id = output[0]
            self.label = int(output[1])
            self.confidence = output[2]
            self.position = np.array((output[3], output[4])) # (x, y)
            self.size = np.array((output[5], output[6])) # (w, h)

        def rescale_roi(self, roi_scale_factor=1.0):
            self.position -= self.size * 0.5 * (roi_scale_factor - 1.0)
            self.size *= roi_scale_factor

        def resize_roi(self, frame_width, frame_height):
            self.position[0] *= frame_width
            self.position[1] *= frame_height
            self.size[0] = self.size[0] * frame_width - self.position[0]
            self.size[1] = self.size[1] * frame_height - self.position[1]

        def clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.position[:] = clip(self.position, min, max)
            self.size[:] = clip(self.size, min, max)

    def __init__(self, model, confidence_threshold=0.5, roi_scale_factor=1.15):
        super(FaceDetector, self).__init__(model)

        assert len(model.input_info) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.input_info))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.input_info[self.input_blob].input_data.shape
        self.output_shape = model.outputs[self.output_blob].shape

        assert len(self.output_shape) == 4 and \
               self.output_shape[3] == self.Result.OUTPUT_SIZE, \
            "Expected model output shape with %s outputs" % \
            (self.Result.OUTPUT_SIZE)

        assert 0.0 <= confidence_threshold and confidence_threshold <= 1.0, \
            "Confidence threshold is expected to be in range [0; 1]"
        self.confidence_threshold = confidence_threshold

        assert 0.0 < roi_scale_factor, "Expected positive ROI scale factor"
        self.roi_scale_factor = roi_scale_factor

    def preprocess(self, frame):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        assert frame.shape[0] == 1
        assert frame.shape[1] == 3
        input = resize_input(frame, self.input_shape)
        return input

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        return super(FaceDetector, self).enqueue({self.input_blob: input})

    def get_roi_proposals(self, frame):
        outputs = self.get_outputs()[0][self.output_blob].buffer
        # outputs shape is [N_requests, 1, 1, N_max_faces, 7]

        frame_width = frame.shape[-1]
        frame_height = frame.shape[-2]

        results = []
        for output in outputs[0][0]:
            result = FaceDetector.Result(output)
            if result.confidence < self.confidence_threshold:
                break # results are sorted by confidence decrease

            result.resize_roi(frame_width, frame_height)
            result.rescale_roi(self.roi_scale_factor)
            result.clip(frame_width, frame_height)

            results.append(result)

        return results
