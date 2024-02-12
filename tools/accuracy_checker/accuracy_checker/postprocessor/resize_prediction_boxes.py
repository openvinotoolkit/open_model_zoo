"""
Copyright (c) 2018-2024 Intel Corporation

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
from ..config import BoolField


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
            ),
            'unpadding': BoolField(
                optional=True, default=False, description='remove padding effect for normalized coordinates'
            ),
            'uncrop': BoolField(
                optional=True, default=False, description='remove center crop effect for normalized coordinates'
            )
        })
        return params

    def configure(self):
        self.rescale = self.get_value_from_config('rescale')
        self.unpadding = self.get_value_from_config('unpadding')
        self.uncrop = self.get_value_from_config('uncrop')

    def process_image(self, annotation, prediction):
        h, w, _ = self.image_size
        for pred in prediction:
            pred.x_mins *= w
            pred.x_maxs *= w
            pred.y_mins *= h
            pred.y_maxs *= h

        return annotation, prediction

    def process_image_with_metadata(self, annotation, prediction, image_metadata=None):
        h, w, _ = self.image_size
        if self.rescale:
            input_h, input_w, _ = image_metadata.get('image_info', self.image_size)
            w = self.image_size[1] / input_w
            h = self.image_size[0] / input_h

        dw = 0
        dh = 0
        if self.uncrop:
            crop_op = [op for op in image_metadata['geometric_operations'] if op.type == 'crop']
            if crop_op:
                dh = (h - min(h, w)) / 2
                dw = (w - min(h, w)) / 2
                h = w = min(h, w)

        if self.unpadding:
            padding_op = [op for op in image_metadata['geometric_operations'] if op.type == 'padding']
            if padding_op:
                padding_op = padding_op[0]
                top, left, bottom, right = padding_op.parameters['pad']
                pw = padding_op.parameters['pref_width']
                ph = padding_op.parameters['pref_height']

                w = pw / (pw - right - left) * self.image_size[1]
                h = ph / (ph - top - bottom) * self.image_size[0]
                dw = left / (pw - right - left) * self.image_size[1]
                dh = top / (ph - top - bottom) * self.image_size[1]


        for pred in prediction:
            pred.x_mins *= w
            pred.x_maxs *= w
            pred.y_mins *= h
            pred.y_maxs *= h
            pred.x_mins += dw
            pred.x_maxs += dw
            pred.y_mins += dh
            pred.y_maxs += dh

        return annotation, prediction
