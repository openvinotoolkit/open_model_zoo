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

from ..postprocessor.postprocessor import Postprocessor
from ..representation import HandLandmarksPrediction, HandLandmarksAnnotation


class HandLandmarksPostprocessor(Postprocessor):
    __provider__ = 'hand_landmarks'

    annotation_types = (HandLandmarksAnnotation, )
    prediction_types = (HandLandmarksPrediction, )

    def process_image(self, annotation, prediction):
        raise ValueError('Postprocessor {} requires image metadata.'.format(self.__provider__))

    def process_image_with_metadata(self, annotation, prediction, image_metadata=None):
        for annotation_, prediction_ in zip(annotation, prediction):
            x_start, y_start, _, _ = annotation_.metadata.get('rect')
            scale_x = image_metadata.get('scale_x')
            scale_y = image_metadata.get('scale_y')

            points = prediction_.x_values.shape[0]
            for i in range(points):
                prediction_.x_values[i] = prediction_.x_values[i] / scale_x + x_start
                prediction_.y_values[i] = prediction_.y_values[i] / scale_y + y_start

        return annotation, prediction
