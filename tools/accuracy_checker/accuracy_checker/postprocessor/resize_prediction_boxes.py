"""
Copyright (c) 2018-2020 Intel Corporation

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
from  ..config import BoolField


class ResizePredictionBoxes(Postprocessor):
    """
    Resize normalized predicted bounding boxes coordinates (i.e. from [0, 1] range) to input image shape.
    """

    __provider__ = 'resize_prediction_boxes'

    prediction_types = (DetectionPrediction, )
    annotation_types = (DetectionAnnotation, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'rescale': BoolField(
                optional=True, default=False, description='required rescale boxes on input size or not'
            )
        })
        return params

    def configure(self):
        self.rescale = self.get_value_from_config('rescale')

    def process_image(self, annotations, predictions):
        h, w, _ = self.image_size
        for prediction in predictions:
            prediction.x_mins *= w
            prediction.x_maxs *= w
            prediction.y_mins *= h
            prediction.y_maxs *= h

        return annotations, predictions

    def process_image_with_metadata(self, annotation, prediction, image_metadata=None):
        h, w, _ = self.image_size
        if self.rescale:
            input_h, input_w, _ = image_metadata.get('image_info', self.image_size)
            w = self.image_size[1] / input_w
            h = self.image_size[0] / input_h

        for pred in prediction:
            pred.x_mins *= w
            pred.x_maxs *= w
            pred.y_mins *= h
            pred.y_maxs *= h

        return annotation, prediction
