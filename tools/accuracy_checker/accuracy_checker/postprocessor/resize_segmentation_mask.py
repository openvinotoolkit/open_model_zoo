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

from functools import singledispatch
import numpy as np
from PIL import Image

from ..config import NumberField, BoolField
from ..utils import get_size_from_config
from .postprocessor import PostprocessorWithSpecificTargets
from ..representation import SegmentationPrediction, SegmentationAnnotation


class ResizeSegmentationMask(PostprocessorWithSpecificTargets):
    __provider__ = 'resize_segmentation_mask'

    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )

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
            'to_dst_image_size': BoolField(optional=True, default=False)
        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)
        self.to_dst_image_size = self.get_value_from_config('to_dst_image_size')

    def process_image(self, annotation, prediction):
        target_height = self.dst_height or self.image_size[0]
        target_width = self.dst_width or self.image_size[1]

        @singledispatch
        def resize_segmentation_mask(entry, height, width):
            return entry

        @resize_segmentation_mask.register(SegmentationPrediction)
        def _(entry, height, width):
            if len(entry.mask.shape) == 2:
                entry.mask = self.segm_resize(entry.mask, width, height)
                return entry

            entry_mask = []
            for class_mask in entry.mask:
                resized_mask = self.segm_resize(class_mask, width, height)
                entry_mask.append(resized_mask)
            entry.mask = np.array(entry_mask)

            return entry

        @resize_segmentation_mask.register(SegmentationAnnotation)
        def _(entry, height, width):
            entry.mask = self.segm_resize(entry.mask, width, height)
            return entry

        for target in annotation:
            resize_segmentation_mask(target, target_height, target_width)

        for target in prediction:
            resize_segmentation_mask(target, target_height, target_width)

        return annotation, prediction

    @staticmethod
    def segm_resize(mask, width, height):
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

    def process_image_with_metadata(self, annotations, predictions, image_metadata=None):
        if 'image_info' in image_metadata and self.to_dst_image_size:
            self.image_size = image_metadata['image_info']
        self.process_image(annotations, predictions)
