"""
 Copyright (c) 2022 Intel Corporation
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
from typing import Any, Dict, Optional

from .segmentation import SegmentationModel
from .types import NumericalValue
from ..adapters.model_adapter import ModelAdapter
from .utils import check_input_parameters_type, create_hard_prediction_from_soft_prediction


class BlurSegmentation(SegmentationModel):
    __model__ = 'blur_segmentation'

    @check_input_parameters_type()
    def __init__(self, model_adapter: ModelAdapter, configuration: Optional[dict] = None, preload: bool = False):
        super().__init__(model_adapter, configuration, preload)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'soft_threshold': NumericalValue(default_value=0.5, min=0.0, max=1.0),
            'blur_strength': NumericalValue(value_type=int, default_value=1, min=0, max=25)
        })

        return parameters

    def _check_io_number(self, number_of_inputs, number_of_outputs):
        pass

    def _get_outputs(self):
        layer_name = 'output'
        layer_shape = self.outputs[layer_name].shape

        if len(layer_shape) == 3:
            self.out_channels = 0
        elif len(layer_shape) == 4:
            self.out_channels = layer_shape[1]
        else:
            raise Exception("Unexpected output layer shape {}. Only 4D and 3D output layers are supported".format(layer_shape))

        return layer_name

    @check_input_parameters_type()
    def postprocess(self, outputs: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        predictions = outputs[self.output_blob_name].squeeze()
        soft_prediction = np.transpose(predictions, axes=(1, 2, 0))
        feature_vector = outputs.get('repr_vector', None)  # Optional output

        hard_prediction = create_hard_prediction_from_soft_prediction(
            soft_prediction=soft_prediction,
            soft_threshold=self.soft_threshold,
            blur_strength=self.blur_strength
        )
        hard_prediction = cv2.resize(hard_prediction, metadata['original_shape'][1::-1], 0, 0, interpolation=cv2.INTER_NEAREST)
        soft_prediction = cv2.resize(soft_prediction, metadata['original_shape'][1::-1], 0, 0, interpolation=cv2.INTER_NEAREST)

        metadata['soft_predictions'] = soft_prediction
        metadata['feature_vector'] = feature_vector

        return hard_prediction
