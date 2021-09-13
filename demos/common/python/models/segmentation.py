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

from .image_model import ImageModel
from .utils import load_labels


class SegmentationModel(ImageModel):
    def __init__(self, ie, model_path, input_transform=None, resize_type='standart',
                 labels=None):
        super().__init__(ie, model_path, input_transform=input_transform, resize_type=resize_type)
        if isinstance(labels, (list, tuple)):
            self.labels = labels
        else:
            self.labels = load_labels(labels) if labels else None

        self.output_blob_name = self._get_outputs()

    def _get_outputs(self):
        if len(self.net.outputs) != 1:
            raise RuntimeError("The Segmentation model wrapper supports topologies only with 1 output")

        blob_name = next(iter(self.net.outputs))
        blob = self.net.outputs[blob_name]

        out_size = blob.shape
        if len(out_size) == 3:
            self.out_channels = 0
        elif len(out_size) == 4:
            self.out_channels = out_size[1]
        else:
            raise Exception("Unexpected output blob shape {}. Only 4D and 3D output blobs are supported".format(out_size))

        return blob_name

    def postprocess(self, outputs, meta):
        predictions = outputs[self.output_blob_name].squeeze()
        input_image_height = meta['original_shape'][0]
        input_image_width = meta['original_shape'][1]

        if self.out_channels < 2: # assume the output is already ArgMax'ed
            result = predictions.astype(np.uint8)
        else:
            result = np.argmax(predictions, axis=0).astype(np.uint8)

        result = cv2.resize(result, (input_image_width, input_image_height), 0, 0, interpolation=cv2.INTER_NEAREST)
        return result


class SalientObjectDetectionModel(SegmentationModel):

    def postprocess(self, outputs, meta):
        input_image_height = meta['original_shape'][0]
        input_image_width = meta['original_shape'][1]
        result = outputs[self.output_blob_name].squeeze()
        result = 1/(1 + np.exp(-result))
        result = cv2.resize(result, (input_image_width, input_image_height), 0, 0, interpolation=cv2.INTER_NEAREST)
        return result
