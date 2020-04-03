"""
Copyright (c) 2019 Intel Corporation

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

from ..representation import DetectionPrediction, DetectionAnnotation
from ..postprocessor.postprocessor import Postprocessor
from ..config import NumberField
from ..utils import get_size_from_config


class RescalePredictionBox(Postprocessor):
    """
    Rescale prediction boxes to original size
    """

    __provider__ = "rescale_prediction_box"

    prediction_types = (DetectionPrediction, )
    annotation_types = (DetectionAnnotation, )

    def process_image(self, annotations, predictions):
        original_height, original_width, _ = self.image_size

        for prediction in predictions:
            image_height, image_width, _ = prediction.metadata.get('image_info')
            prediction.x_mins *= original_width/image_width
            prediction.x_maxs *= original_width/image_width
            prediction.y_mins *= original_height/image_height
            prediction.y_maxs *= original_height/image_height

        return annotations, predictions
