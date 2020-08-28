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

from .postprocessor import PostprocessorWithSpecificTargets, Postprocessor
from ..representation import (
    BrainTumorSegmentationAnnotation, BrainTumorSegmentationPrediction, SegmentationAnnotation, SegmentationPrediction
)
from ..config import NumberField
from ..preprocessor import Crop3D, CropOrPad
from ..utils import get_size_3d_from_config, get_size_from_config


class CropSegmentationMask(PostprocessorWithSpecificTargets):
    __provider__ = 'crop_segmentation_mask'

    annotation_types = (BrainTumorSegmentationAnnotation,)
    prediction_types = (BrainTumorSegmentationPrediction,)

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
            'dst_volume': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination volume for mask cropping."
            ),
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for mask cropping for both dimensions."
            )
        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width, self.dst_volume = get_size_3d_from_config(self.config)

    def process_image(self, annotation, prediction):
        for target in annotation:
            target.mask = Crop3D.crop_center(target.mask, self.dst_height, self.dst_width, self.dst_volume)

        for target in prediction:
            target.mask = Crop3D.crop_center(target.mask, self.dst_height, self.dst_width, self.dst_volume)

        return annotation, prediction


class CropOrPadSegmentationMask(Postprocessor):
    __provider__ = 'crop_or_pad'

    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
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
        return params

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config)

    def process_image(self, annotation, prediction):
        for ann in annotation:
            ann.mask = self.process_mask(ann.mask)
        return annotation, prediction

    def process_mask(self, mask):
        if len(mask.shape) == 2:
            height, width = mask.shape
        else:
            height, width, _ = mask.shape

        width_diff = self.dst_width - width
        offset_crop_width = max(-width_diff // 2, 0)
        offset_pad_width = max(width_diff // 2, 0)

        height_diff = self.dst_height - height
        offset_crop_height = max(-height_diff // 2, 0)
        offset_pad_height = max(height_diff // 2, 0)
        cropped, _ = CropOrPad.crop_to_bounding_box(
            mask, offset_crop_height, offset_crop_width, min(self.dst_height, height), min(self.dst_width, width))
        resized, _ = CropOrPad.pad_to_bounding_box(
            cropped, offset_pad_height, offset_pad_width, self.dst_height, self.dst_width
        )
        return resized
