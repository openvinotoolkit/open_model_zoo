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
from .utils import Detection, resize_image, load_labels, clip_detections


class DETR(Model):
    def __init__(self, ie, model_path, input_transform, labels=None, threshold=0.5):
        super().__init__(ie, model_path, input_transform)

        assert len(self.net.input_info) == 1, "Expected 1 input blob"
        self.image_blob_name = next(iter(self.net.input_info))

        assert len(self.net.outputs) == 2, "Expected 2 output blobs"
        self.bboxes_blob_name, self.scores_blob_name = self._parse_outputs()

        if isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            self.labels = load_labels(labels) if labels else None

        self.threshold = threshold

        self.n, self.c, self.h, self.w = self.net.input_info[self.image_blob_name].input_data.shape
        assert self.c == 3, "Expected 3-channel input"

    def _parse_outputs(self):
        (bboxes_blob_name, bboxes_layer), (scores_blob_name, scores_layer) = self.net.outputs.items()

        assert bboxes_layer.shape[1] == scores_layer.shape[1], "Expected the same dimension for boxes and scores"

        if bboxes_layer.shape[2] == 4:
            return bboxes_blob_name, scores_blob_name
        elif scores_layer.shape[2] == 4:
            return scores_blob_name, bboxes_blob_name
        else:
            raise RuntimeError("Expected shape [:,:,4] for bboxes output, but got {} and {}"
                               .format(*[output.shape for output in self.net.outputs]))

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

        x_mins, y_mins, x_maxs, y_maxs = self.box_cxcywh_to_xyxy(boxes)

        x_mins = x_mins * meta['original_shape'][1]
        y_mins = y_mins * meta['original_shape'][0]
        x_maxs = x_maxs * meta['original_shape'][1]
        y_maxs = y_maxs * meta['original_shape'][0]

        scores = self.softmax(scores)
        labels = np.argmax(scores[:, :-1], axis=-1)
        det_scores = np.max(scores[:, :-1], axis=-1)

        keep = det_scores > self.threshold

        detections = [Detection(*det) for det in zip(x_mins[keep], y_mins[keep], x_maxs[keep], y_maxs[keep],
                                                     det_scores[keep], labels[keep])]
        return clip_detections(detections, meta['original_shape'])

    @staticmethod
    def box_cxcywh_to_xyxy(box):
        x_c, y_c, w, h = box.T
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return b

    @staticmethod
    def softmax(logits):
        res = [np.exp(logit) / np.sum(np.exp(logit)) for logit in logits]
        return np.array(res)
