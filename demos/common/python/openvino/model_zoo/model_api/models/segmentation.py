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
from .types import ListValue, StringValue
from .utils import load_labels


class SegmentationModel(ImageModel):
    __model__ = 'Segmentation'

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self._check_io_number(1, 1)
        if self.path_to_labels:
            self.labels = load_labels(self.path_to_labels)

        self.output_blob_name = self._get_outputs()

    def _get_outputs(self):
        layer_name = next(iter(self.outputs))
        layer_shape = self.outputs[layer_name].shape

        if len(layer_shape) == 3:
            self.out_channels = 0
        elif len(layer_shape) == 4:
            self.out_channels = layer_shape[1]
        else:
            raise Exception("Unexpected output layer shape {}. Only 4D and 3D output layers are supported".format(layer_shape))

        return layer_name

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'labels': ListValue(description="List of class labels"),
            'path_to_labels': StringValue(description="Path to file with labels. Overrides the labels, if they sets via 'labels' parameter")
        })

        return parameters

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
    __model__ = 'Salient_Object_Detection'

    def postprocess(self, outputs, meta):
        input_image_height = meta['original_shape'][0]
        input_image_width = meta['original_shape'][1]
        result = outputs[self.output_blob_name].squeeze()
        result = 1/(1 + np.exp(-result))
        result = cv2.resize(result, (input_image_width, input_image_height), 0, 0, interpolation=cv2.INTER_NEAREST)
        return result
