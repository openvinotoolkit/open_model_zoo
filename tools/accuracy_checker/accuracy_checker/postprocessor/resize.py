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

from functools import singledispatch
from PIL import Image
import numpy as np
import cv2
from ..representation import (
    SegmentationPrediction, SegmentationAnnotation,
    StyleTransferAnnotation, StyleTransferPrediction,
    SuperResolutionPrediction, SuperResolutionAnnotation,
    ImageProcessingPrediction, ImageProcessingAnnotation,
    ImageInpaintingAnnotation, ImageInpaintingPrediction,
    SalientRegionAnnotation, SalientRegionPrediction,
    BackgroundMattingAnnotation, BackgroundMattingPrediction
)
from ..postprocessor.postprocessor import PostprocessorWithSpecificTargets, ApplyToOption
from ..postprocessor import ResizeSegmentationMask
from ..config import NumberField, StringField
from ..utils import get_size_from_config


class Resize(PostprocessorWithSpecificTargets):

    __provider__ = 'resize'

    prediction_types = (
        StyleTransferPrediction, ImageProcessingPrediction,
        SegmentationPrediction, SuperResolutionPrediction,
        ImageInpaintingPrediction, SalientRegionPrediction,
        BackgroundMattingPrediction
    )
    annotation_types = (
        StyleTransferAnnotation, ImageProcessingAnnotation,
        SegmentationAnnotation, SuperResolutionAnnotation,
        ImageInpaintingPrediction, SalientRegionAnnotation,
        BackgroundMattingAnnotation
    )

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
            ),
            'resize_realization': StringField(
                optional=True, choices=["pillow", "opencv"], default="pillow",
                description="Parameter specifies functionality of which library will be used for resize: "
                            "{}".format(', '.join(["pillow", "opencv"]))
            ),

        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)
        self._required_both = True
        self.realization = self.get_value_from_config('resize_realization')

    def process_image_with_metadata(self, annotation, prediction, image_metadata=None):
        if self._deprocess_predictions:
            self._calculate_scale(image_metadata)
        self.process_image(annotation, prediction)

    def _calculate_scale(self, image_metadata):
        if image_metadata is None:
            self.x_scale, self.y_scale = 1, 1
            return
        image_h, image_w = image_metadata['image_size'][:2]
        input_shape = next(iter(image_metadata['input_shape'].values()))
        input_h, input_w = input_shape[2:] if input_shape[1] in [1, 3, 4] else input_shape[1:3]
        self.x_scale = image_w / input_w
        self.y_scale = image_h / input_h

    def process_image(self, annotations, predictions):
        @singledispatch
        def resize(entry, height, width):
            return entry

        @resize.register(StyleTransferAnnotation)
        @resize.register(StyleTransferPrediction)
        @resize.register(SuperResolutionAnnotation)
        @resize.register(SuperResolutionPrediction)
        @resize.register(ImageProcessingAnnotation)
        @resize.register(ImageProcessingPrediction)
        @resize.register(ImageInpaintingAnnotation)
        @resize.register(ImageInpaintingPrediction)
        def _(entry, height, width):
            data = entry.value if entry.value.shape[-1] > 1 else entry.value[:, :, 0]
            assert self.realization in ['pillow', 'opencv']
            if self.realization == 'pillow':
                data = data.astype(np.uint8)
                data = Image.fromarray(data)
                data = data.resize((width, height), Image.BICUBIC)
            else:
                data = cv2.resize(data, (width, height)).astype(np.uint8)

            entry.value = np.array(data)
            return entry

        @resize.register(SegmentationPrediction)
        @resize.register(SalientRegionPrediction)
        @resize.register(BackgroundMattingPrediction)
        def _(entry, height, width):
            if len(entry.mask.shape) == 2:
                entry.mask = ResizeSegmentationMask.segm_resize(entry.mask, width, height)
                return entry

            entry_mask = []
            for class_mask in entry.mask:
                resized_mask = ResizeSegmentationMask.segm_resize(class_mask, width, height)
                entry_mask.append(resized_mask)
            entry.mask = np.array(entry_mask)

            return entry

        @resize.register(SegmentationAnnotation)
        @resize.register(SalientRegionAnnotation)
        @resize.register(BackgroundMattingAnnotation)
        def _(entry, height, width):
            entry.mask = ResizeSegmentationMask.segm_resize(entry.mask, width, height)

            return entry

        @singledispatch
        def set_sizes(entry):
            height = self.dst_height if self.dst_height else self.image_size[0]
            width = self.dst_width if self.dst_width else self.image_size[1]

            return height, width

        @set_sizes.register(SuperResolutionAnnotation)
        def _(entry):
            height = self.dst_height if self.dst_height else entry.value.shape[0]
            width = self.dst_width if self.dst_width else entry.value.shape[1]

            return height, width

        @set_sizes.register(SuperResolutionPrediction)
        def _(entry):
            if self._deprocess_predictions:
                height = int(entry.value.shape[0] * self.y_scale)
                width = int(entry.value.shape[1] * self.x_scale)
                return height, width
            height = self.dst_height if self.dst_height else entry.value.shape[0]
            width = self.dst_width if self.dst_width else entry.value.shape[1]

            return height, width

        @set_sizes.register(SegmentationPrediction)
        @set_sizes.register(SalientRegionPrediction)
        @set_sizes.register(BackgroundMattingPrediction)
        def _(entry):
            if self._deprocess_predictions:
                return self.image_size[:2]
            height = self.dst_height if self.dst_height else self.image_size[0]
            width = self.dst_width if self.dst_width else self.image_size[1]

            return height, width

        if self.apply_to is None or self.apply_to in [ApplyToOption.PREDICTION, ApplyToOption.ALL]:
            if annotations:
                for annotation, prediction in zip(annotations, predictions):
                    height, width = set_sizes(annotation or prediction)
                    resize(prediction, height, width)
            else:
                for prediction in predictions:
                    height, width = set_sizes(prediction)
                    resize(prediction, height, width)

        if self.apply_to is None or self.apply_to in [ApplyToOption.ANNOTATION, ApplyToOption.ALL]:
            for annotation in annotations:
                if annotation is None:
                    continue
                height, width = set_sizes(annotation)
                resize(annotation, height, width)

        return annotations, predictions
