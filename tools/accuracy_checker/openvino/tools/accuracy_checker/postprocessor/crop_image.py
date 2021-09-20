"""
Copyright (c) 2018-2021 Intel Corporation

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

from ..config import NumberField, StringField
from ..utils import get_size_from_config
from .postprocessor import PostprocessorWithSpecificTargets
from ..preprocessor import Crop, CornerCrop
from ..representation import (
    ImageProcessingAnnotation,
    ImageProcessingPrediction,
    ImageInpaintingAnnotation,
    ImageInpaintingPrediction
)


class CropImage(PostprocessorWithSpecificTargets):
    __provider__ = "crop_image"

    annotation_types = (ImageInpaintingAnnotation, ImageProcessingAnnotation)
    prediction_types = (ImageInpaintingPrediction, ImageProcessingPrediction)

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
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)

    def process_image(self, annotation, prediction):
        target_height = self.dst_height or self.image_size[0]
        target_width = self.dst_width or self.image_size[1]

        for target in annotation:
            target.value = Crop.process_data(
                target.value, target_height, target_width, None, False, False, True, {}
            )

        for target in prediction:
            target.value = Crop.process_data(
                target.value, target_height, target_width, None, False, False, True, {}
            )

        return annotation, prediction

    def process_image_with_metadata(self, annotations, predictions, image_metadata=None):
        if 'image_size' in image_metadata:
            self.image_size = image_metadata['image_size']
        self.process_image(annotations, predictions)


class CornerCropImage(PostprocessorWithSpecificTargets):
    __provider__ = 'corner_crop_image'

    annotation_types = (ImageInpaintingAnnotation, ImageProcessingAnnotation)
    prediction_types = (ImageInpaintingPrediction, ImageProcessingPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for crop."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for crop."
            ),
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for crop for both dimensions (height and width)."
            ),
            'corner_type': StringField(
                optional=True, choices=['top_left', 'top_right', 'bottom_left', 'bottom_right'],
                default='top_left', description="Destination height for image cropping respectively."
            ),
        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)
        self.corner_type = self.get_value_from_config('corner_type')

    def process_image(self, annotation, prediction):
        target_height = self.dst_height or self.image_size[0]
        target_width = self.dst_width or self.image_size[1]

        for target in annotation:
            target.value = CornerCrop.process_data(target.value, target_height, target_width, self.corner_type)

        for target in prediction:
            target.value = CornerCrop.process_data(target.value, target_height, target_width, self.corner_type)

        return annotation, prediction

    def process_image_with_metadata(self, annotations, predictions, image_metadata=None):
        if 'image_size' in image_metadata:
            self.image_size = image_metadata['image_size']
        self.process_image(annotations, predictions)
