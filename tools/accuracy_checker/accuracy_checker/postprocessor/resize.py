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

from functools import singledispatch
import cv2
from PIL import Image
import numpy as np
from ..representation import (
    DepthEstimationAnnotation, DepthEstimationPrediction,
    SegmentationPrediction, SegmentationAnnotation,
    StyleTransferAnnotation, StyleTransferPrediction,
    SuperResolutionPrediction, SuperResolutionAnnotation)
from ..postprocessor.postprocessor import PostprocessorWithSpecificTargets
from ..postprocessor import ResizeSegmentationMask
from ..config import NumberField
from ..utils import get_size_from_config


class Resize(PostprocessorWithSpecificTargets):

    __provider__ = 'resize'

    prediction_types = (DepthEstimationPrediction, StyleTransferPrediction,
                        SegmentationPrediction, SuperResolutionPrediction, )
    annotation_types = (DepthEstimationAnnotation, StyleTransferAnnotation,
                        SegmentationAnnotation, SuperResolutionAnnotation, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for resize"
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for resize."
            ),
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for resize for both dimensions (height and width)."
            )
        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)

    def process_image(self, annotations, predictions):
        @singledispatch
        def resize(entry, height, width):
            return entry

        @resize.register(DepthEstimationPrediction)
        def _(entry, height, width):
            if not self.dst_height:
                height = self.image_size[0]
            if not self.dst_width:
                width = self.image_size[1]
            entry.depth_map = cv2.resize(entry.depth_map, (width, height))

            return entry

        @resize.register(StyleTransferAnnotation)
        @resize.register(SuperResolutionPrediction)
        def _(entry, height, width):
            data = Image.fromarray(entry.value)
            data = data.resize((width, height), Image.BICUBIC)
            entry.value = np.array(data)

            return entry

        @resize.register(SegmentationPrediction)
        def _(entry, height, width):
            if not self.dst_height:
                height = self.image_size[0]
            if not self.dst_width:
                width = self.image_size[1]
            if len(entry.mask.shape) == 2:
                entry.mask = ResizeSegmentationMask(self.config).segm_resize(entry.mask, width, height)
                return entry

            entry_mask = []
            for class_mask in entry.mask:
                resized_mask = ResizeSegmentationMask(self.config).segm_resize(class_mask, width, height)
                entry_mask.append(resized_mask)
            entry.mask = np.array(entry_mask)

            return entry

        @resize.register(SegmentationAnnotation)
        def _(entry, height, width):
            if not self.dst_height:
                height = self.image_size[0]
            if not self.dst_width:
                width = self.image_size[1]
            entry.mask = ResizeSegmentationMask.segm_resize(entry.mask, width, height)

            return entry

        @singledispatch
        def set_sizes(entry):
            return self.dst_height, self.dst_width

        @set_sizes.register(StyleTransferAnnotation)
        @set_sizes.register(SuperResolutionAnnotation)
        def _(entry):
            height = self.dst_height if self.dst_height else entry.shape[0]
            width = self.dst_width if self.dst_width else entry.shape[1]

            return height, width

        for annotation, prediction in zip(annotations, predictions):
            height, width = set_sizes(annotation)
            resize(prediction, height, width)

        for annotation in annotations:
            height, width = set_sizes(annotation)
            resize(annotation, height, width)

        return annotations, predictions
