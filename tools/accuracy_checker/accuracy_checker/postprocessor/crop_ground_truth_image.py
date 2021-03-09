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

from .postprocessor import Postprocessor
from ..preprocessor import Crop, CornerCrop
from ..representation import ImageInpaintingAnnotation, ImageInpaintingPrediction
from ..config import NumberField, StringField
from ..utils import get_size_from_config


class CropGTImage(Postprocessor):
    __provider__ = "crop_ground_truth_image"

    annotation_types = (ImageInpaintingAnnotation,)
    prediction_types = (ImageInpaintingPrediction,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for mask cropping"
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for mask cropping."
            ),
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for mask cropping for both dimensions."
            )
        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config)

    def process_image(self, annotation, prediction):
        for target in annotation:
            target.value = Crop.process_data(
                target.value, self.dst_height, self.dst_width, None, False, False, True, {}
            )

        return annotation, prediction

class CornerCropGTImage(Postprocessor):
    __provider__ = "corner_crop_ground_truth_image"

    annotation_types = (ImageInpaintingAnnotation,)
    prediction_types = (ImageInpaintingPrediction,)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for mask cropping."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for mask cropping."
            ),
            'corner_type': StringField(
                optional=True, choices=['top_left', 'top_right', 'bottom_left', 'bottom_right'],
                default='top_left', description="Destination height for image cropping respectively."
            ),
        })
        return parameters

    def configure(self):
        self.corner_type = self.get_value_from_config('corner_type')
        self.dst_height, self.dst_width = get_size_from_config(self.config)

    def process_image(self, annotation, prediction):
        for target in annotation:
            target.value = CornerCrop.process_data(target.value, self.dst_height, self.dst_width, self.corner_type)

        return annotation, prediction
