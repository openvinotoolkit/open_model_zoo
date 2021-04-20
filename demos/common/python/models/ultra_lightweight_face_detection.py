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
import numpy as np

from .model import Model
from .utils import Detection, resize_image, nms


class UltraLightweightFaceDetection(Model):
    def __init__(self, ie, model_path, input_transform, threshold=0.5):
        super().__init__(ie, model_path, input_transform)

        assert len(self.net.input_info) == 1, "Expected 1 input blob"
        self.image_blob_name = next(iter(self.net.input_info))

        assert len(self.net.outputs) == 2, "Expected 2 output blobs"
        self.bboxes_blob_name, self.scores_blob_name = self._parse_outputs()

        self.labels = ['Face']

        self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
        assert self.c == 3, "Expected 3-channel input"

        self.confidence_threshold = threshold
        self.nms_threshold = 0.5

    def _parse_outputs(self):
        bboxes_blob_name = None
        scores_blob_name = None
        for name, layer in self.net.outputs.items():
            if layer.shape[2] == 4:
                bboxes_blob_name = name
            elif layer.shape[2] == 2:
                scores_blob_name = name
            else:
                raise RuntimeError("Expected shapes [:,:,4] and [:,:2] for outputs, but got {} and {}"
                                   .format(*[output.shape for output in self.net.outputs]))
        assert self.net.outputs[bboxes_blob_name].shape[1] == self.net.outputs[scores_blob_name].shape[1], \
            "Expected the same dimension for boxes and scores"
        return bboxes_blob_name, scores_blob_name

    def preprocess(self, inputs):
        image = inputs

        resized_image = resize_image(image, (self.w, self.h))
        meta = {'original_shape': image.shape,
                'resized_shape': resized_image.shape}
        resized_image = self.input_transform(resized_image)
        resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        resized_image = resized_image.reshape((self.n, self.c, self.h, self.w))

        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, meta

    def postprocess(self, outputs, meta):
        boxes = outputs[self.bboxes_blob_name][0]
        scores = outputs[self.scores_blob_name][0]

        score = np.transpose(scores)[1]

        mask = score > self.confidence_threshold
        filtered_boxes, filtered_score = boxes[mask, :], score[mask]

        x_mins, y_mins, x_maxs, y_maxs = filtered_boxes.T

        keep = nms(x_mins, y_mins, x_maxs, y_maxs, filtered_score, self.nms_threshold)

        filtered_score = filtered_score[keep]
        x_mins = x_mins[keep] * meta['original_shape'][1]
        y_mins = y_mins[keep] * meta['original_shape'][0]
        x_maxs = x_maxs[keep] * meta['original_shape'][1]
        y_maxs = y_maxs[keep] * meta['original_shape'][0]

        return [Detection(*det, 0) for det in zip(x_mins, y_mins, x_maxs, y_maxs, filtered_score)]
