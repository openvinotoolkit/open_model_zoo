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
import numpy as np
from ..representation import (
    DetectionPrediction, DetectionAnnotation,
    DepthEstimationAnnotation, DepthEstimationPrediction,
    SegmentationPrediction, SegmentationAnnotation,
    StyleTransferAnnotation, StyleTransferPrediction,
    SuperResolutionPrediction, SuperResolutionAnnotation)
from ..postprocessor.postprocessor import PostprocessorWithSpecificTargets
from ..config import NumberField
from ..utils import get_size_from_config

try:
    from PIL import Image
except ImportError:
    Image = None


class Resize(PostprocessorWithSpecificTargets):

    __provider__ = 'resize'

    prediction_types = (DetectionPrediction, DepthEstimationPrediction,
                        StyleTransferPrediction, SegmentationPrediction,
                        SuperResolutionPrediction, )
    annotation_types = (DetectionAnnotation, DepthEstimationAnnotation,
                        StyleTransferAnnotation, SegmentationAnnotation,
                        SuperResolutionAnnotation, )

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
        if Image is None:
            raise ValueError('{} requires pillow, please install it'.format(self.__provider__))
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)

    def process_image(self, annotations, predictions):
        if self.dst_width and self.dst_height:
            height = self.dst_height
            width = self.dst_width
        else:
            height = self.image_size[0]
            width = self.image_size[1]

        @singledispatch
        def resize(entry, height, width):
            return entry

        @resize.register(DetectionPrediction)
        def _(entry, height, width):
            entry.x_mins *= width
            entry.x_maxs *= width
            entry.y_mins *= height
            entry.y_maxs *= height

            return entry

        @resize.register(DepthEstimationPrediction)
        def _(entry, height, width):
            entry.depth_map = cv2.resize(entry.depth_map, (width, height))

            return entry

        @resize.register(StyleTransferAnnotation)
        def _(entry, height, width):
            data = Image.fromarray(entry.value)
            data = data.resize((width, height), Image.BICUBIC)
            entry.value = np.array(data)

            return entry

        @resize.register(SegmentationPrediction)
        def _(entry, height, width):
            if len(entry.mask.shape) == 2:
                entry.mask = self._segm_resize(entry.mask, width, height)
                return entry

            entry_mask = []
            for class_mask in entry.mask:
                resized_mask = self._segm_resize(class_mask, width, height)
                entry_mask.append(resized_mask)
            entry.mask = np.array(entry_mask)

            return entry

        @resize.register(SegmentationAnnotation)
        def _(entry, height, width):
            entry.mask = self._segm_resize(entry.mask, width, height)
            return entry

        @resize.register(SuperResolutionPrediction)
        def _(entry, height, width):
            data = Image.fromarray(entry.value)
            data = data.resize((width, height), Image.BICUBIC)
            entry.value = np.array(data)

            return entry

        for prediction in predictions:
            resize(prediction, height, width)

        for annotation in annotations:
            resize(annotation, height, width)

        return annotations, predictions

    def _segm_resize(self, mask, width, height):
        def _to_image(arr):
            data = np.asarray(arr)
            if np.iscomplexobj(data):
                raise ValueError("Cannot convert a complex-valued array.")
            shape = list(data.shape)
            if len(shape) == 2:
                return _process_2d(data, shape)
            if len(shape) == 3 and shape[2] in (3, 4):
                return _process_3d(data, shape)
            raise ValueError("'arr' does not have a suitable array shape for any mode.")

        def _process_2d(data, shape):
            height, width = shape
            bytedata = _bytescale(data)
            image = Image.frombytes('L', (width, height), bytedata.tostring())

            return image

        def _process_3d(data, shape):
            bytedata = _bytescale(data)
            height, width, channels = shape
            mode = 'RGB' if channels == 3 else 'RGBA'
            image = Image.frombytes(mode, (width, height), bytedata.tostring())

            return image

        def _bytescale(data):
            if data.dtype == np.uint8:
                return data
            cmin = data.min()
            cmax = data.max()
            cscale = cmax - cmin
            if cscale == 0:
                cscale = 1

            scale = float(255) / cscale
            bytedata = (data - cmin) * scale

            return (bytedata.clip(0, 255) + 0.5).astype(np.uint8)

        image = _to_image(mask)
        image_new = image.resize((width, height), resample=0)

        return np.array(image_new)
