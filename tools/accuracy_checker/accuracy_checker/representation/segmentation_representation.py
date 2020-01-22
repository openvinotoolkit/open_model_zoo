"""
Copyright (c) 2019 Intel Corporation

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

from enum import Enum

import numpy as np

try:
    import pycocotools.mask as maskUtils
except ImportError:
    maskUtils = None

from .base_representation import BaseRepresentation
from ..data_readers import BaseReader
from ..utils import remove_difficult


class GTMaskLoader(Enum):
    PILLOW = 0
    OPENCV = 1
    SCIPY = 2
    NIFTI = 3
    NUMPY = 4
    NIFTI_CHANNELS_FIRST = 5

LOADERS_MAPPING = {
    'opencv': GTMaskLoader.OPENCV,
    'pillow': GTMaskLoader.PILLOW,
    'scipy': GTMaskLoader.SCIPY,
    'nifti': GTMaskLoader.NIFTI,
    'nifti_channels_first': GTMaskLoader.NIFTI_CHANNELS_FIRST,
    'numpy': GTMaskLoader.NUMPY
}


class SegmentationRepresentation(BaseRepresentation):
    pass


class SegmentationAnnotation(SegmentationRepresentation):
    LOADERS = {
        GTMaskLoader.PILLOW: 'pillow_imread',
        GTMaskLoader.OPENCV: 'opencv_imread',
        GTMaskLoader.SCIPY: 'scipy_imread',
        GTMaskLoader.NIFTI: 'nifti_reader',
        GTMaskLoader.NIFTI_CHANNELS_FIRST: {'type': 'nifti_reader', 'channels_first': True},
        GTMaskLoader.NUMPY: 'numpy_reader'
    }

    def __init__(self, identifier, path_to_mask, mask_loader=GTMaskLoader.PILLOW):
        """
        Args:
            identifier: object identifier (e.g. image name).
            path_to_mask: path where segmentation mask should be loaded from. The path is relative to data source.
            mask_loader: back-end, used to load segmentation masks.
        """

        super().__init__(identifier)
        self._mask_path = path_to_mask
        self._mask_loader = mask_loader
        self._mask = None

    @property
    def mask(self):
        return self._mask if self._mask is not None else self._load_mask()

    @mask.setter
    def mask(self, value):
        self._mask = value

    def _load_mask(self):
        if self._mask is None:
            loader_config = self.LOADERS.get(self._mask_loader)
            if isinstance(loader_config, str):
                loader = BaseReader.provide(loader_config, self.metadata['data_source'])
            else:
                loader = BaseReader.provide(loader_config['type'], self.metadata['data_source'], config=loader_config)
            if self._mask_loader == GTMaskLoader.PILLOW:
                loader.convert_to_rgb = False
            mask = loader.read(self._mask_path)
            return mask.astype(np.uint8)

        return self._mask


class SegmentationPrediction(SegmentationRepresentation):
    def __init__(self, identifiers, mask):
        """
        Args:
            identifiers: object identifier (e.g. image name).
            mask: array with shape (n_classes, height, width) of probabilities at each location.
        """

        super().__init__(identifiers)
        self.mask = mask


class BrainTumorSegmentationAnnotation(SegmentationAnnotation):
    def __init__(self, identifier, path_to_mask, loader=GTMaskLoader.NIFTI, box=None):
        super().__init__(identifier, path_to_mask, loader)
        self.box = box


class BrainTumorSegmentationPrediction(SegmentationPrediction):
    def __init__(self, identifiers, mask, label_order=(0, 1, 2, 3)):
        super().__init__(identifiers, mask)
        self.label_order = label_order


class CoCoInstanceSegmentationRepresentation(SegmentationRepresentation):
    def __init__(self, identifier, mask, labels):
        if not maskUtils:
            raise ValueError('can not create representation')
        super().__init__(identifier)
        self.raw_mask = mask
        self.labels = labels
        self._mask = None

    @property
    def mask(self):
        return self._mask if self._mask is not None else self._load_mask()

    def _load_mask(self):
        masks = []
        image_size = self.metadata['image_size']
        height, width, _ = image_size if len(np.shape(image_size)) == 1 else image_size[0]
        for mask in self.raw_mask:
            converted_mask = self._convert_mask(mask, height, width)
            masks.append(converted_mask)

        self._mask = masks

        return self._mask

    @staticmethod
    def _convert_mask(mask, height, width):
        if maskUtils and isinstance(mask, list):
            rles = maskUtils.frPyObjects(mask, height, width)
            rle = maskUtils.merge(rles)
        elif maskUtils and isinstance(mask['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask, height, width)
        else:
            rle = mask
            # rle
        return rle

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def size(self):
        return len(self.raw_mask)

    @property
    def areas(self):
        precomputed_areas = self.metadata.get('areas')
        if precomputed_areas:
            return precomputed_areas
        masks = self.mask
        areas = []
        for mask in masks:
            areas.append(maskUtils.area(mask))
        return areas


class CoCoInstanceSegmentationAnnotation(CoCoInstanceSegmentationRepresentation):
    pass


class CoCocInstanceSegmentationPrediction(CoCoInstanceSegmentationRepresentation):

    def __init__(self, identifier, mask, labels, scores):
        super().__init__(identifier, mask, labels)

        self.scores = scores

    def remove(self, indexes):
        self.labels = np.delete(self.labels, indexes)

        self.mask = np.delete(self.mask, indexes)

        self.scores = np.delete(self.scores, indexes)

        difficult_boxes = self.metadata.get('difficult_boxes')

        if not difficult_boxes:
            return

        new_difficult_boxes = remove_difficult(difficult_boxes, indexes)

        self.metadata['difficult_boxes'] = new_difficult_boxes
