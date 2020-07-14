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
import os
import numpy as np

from postprocessor import RetinaFacePostprocessor

class Detector(object):
    def __init__(self, ie, model_path, face_prob_threshold, device='CPU'):
        model = ie.read_network(model_path, os.path.splitext(model_path)[0] + '.bin')

        assert len(model.input_info) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 12 or len(model.outputs) == 9, "Expected 12 or 9 output blobs"

        self._input_layer_name = next(iter(model.input_info))
        self._output_layer_names = model.outputs
        _, channels, self.input_height, self.input_width = model.input_info[self._input_layer_name].input_data.shape
        assert channels == 3, "Expected 3-channel input"

        self._detect_masks = len(model.outputs) == 12
        self.face_prob_threshold = face_prob_threshold

        self._ie = ie
        self._exec_model = self._ie.load_network(model, device)

        self.infer_time = -1

    def infer(self, image):
        t0 = cv2.getTickCount()
        output = self._exec_model.infer(inputs={self._input_layer_name: image})
        self.infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
        return output

    def detect(self, image):
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        output = self.infer(image)
        scale_x = self.input_width/width
        scale_y = self.input_height/height
        postprocessor = RetinaFacePostprocessor(self._detect_masks)
        detections = postprocessor.process_output(output, scale_x, scale_y, self.face_prob_threshold)
        return detections, self._detect_masks
