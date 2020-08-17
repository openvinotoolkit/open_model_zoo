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

from enum import Enum
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import warnings
import cv2 as cv

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
            data_source = self.metadata.get('segmentation_masks_source') or self.metadata.get('additional_data_source')
            if data_source is None:
                data_source = self.metadata['data_source']
            if isinstance(loader_config, str):
                loader = BaseReader.provide(loader_config, data_source)
            else:
                loader = BaseReader.provide(loader_config['type'], data_source, config=loader_config)
            if self._mask_loader == GTMaskLoader.PILLOW:
                loader.convert_to_rgb = False
            mask = loader.read(self._mask_path)
            return mask.astype(np.uint8)

        return self._mask

    @staticmethod
    def _encode_mask(mask, segmentation_colors):
        if len(mask.shape) != 3:
            return mask

        mask = mask.astype(int)
        num_channels = len(mask.shape)
        encoded_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for label, color in enumerate(segmentation_colors):
            encoded_mask[np.where(
                np.all(mask == color, axis=-1) if num_channels >= 3 else mask == color
            )[:2]] = label

        return encoded_mask

    def to_polygon(self, segmentation_colors=None, label_map=None):
        if self.mask is None or self.mask.size == 0:
            warnings.warn("Polygon can be found only for non-empty mask")
            return {}

        if self.metadata.get('dataset_meta'):
            if not segmentation_colors and self.metadata['dataset_meta'].get('segmentation_colors'):
                segmentation_colors = self.metadata['dataset_meta']['segmentation_colors']
            if not label_map and self.metadata['dataset_meta'].get('label_map'):
                label_map = self.metadata['dataset_meta']['label_map']

        if not segmentation_colors and len(self.mask.shape) == 3:
            raise ValueError("Mask should be decoded, but there is no segmentation colors")

        mask = self._encode_mask(self.mask, segmentation_colors) if segmentation_colors else self.mask

        polygons = defaultdict(list)
        indexes = np.unique(mask) if not label_map else set(np.unique(mask))&set(label_map.keys())
        for i in indexes:
            binary_mask = np.uint8(mask == i)
            contours, _ = cv.findContours(binary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.size < 6:
                    continue
                contour = np.squeeze(contour, axis=1)
                polygons[i].append(contour)

        return polygons


class SegmentationPrediction(SegmentationRepresentation):
    def __init__(self, identifiers, mask):
        """
        Args:
            identifiers: object identifier (e.g. image name).
            mask: array with shape (n_classes, height, width) of probabilities at each location.
        """

        super().__init__(identifiers)
        self.mask = mask

    def to_annotation(self, **kwargs):
        mask_source = Path.cwd() / 'dumped_masks'
        if not mask_source.exists():
            mask_source.mkdir()
        mask_file = mask_source / '{}'.format(str(self.identifier).split('.')[0] + '.npy')
        mask_shape = self.mask.shape
        if len(mask_shape) == 3 and mask_shape[0] != 1:
            argmaxed_mask = np.argmax(self.mask, axis=0).astype(np.uint8)
            argmaxed_mask.dump(str(mask_file))
        else:
            self.mask.dump()
        annotation_meta = deepcopy(self.metadata or {})
        annotation_meta['data_source'] = mask_file.parent
        annotation = SegmentationAnnotation(self.identifier, mask_file.name, mask_loader=GTMaskLoader.NUMPY)
        annotation.metadata = annotation_meta

        return annotation

    def to_polygon(self):
        if self.mask is None or self.mask.size == 0:
            warnings.warn("Polygon can be found only for non-empty mask")
            return {}

        polygons = defaultdict(list)

        mask = self.mask

        if len(mask.shape) == 3:
            if 1 not in mask.shape:
                mask = np.argmax(mask, axis=0)
            else:
                mask = np.squeeze(mask, axis=-1)
        indexes = np.unique(mask)
        for i in indexes:
            binary_mask = np.uint8(mask == i)
            contours, _ = cv.findContours(binary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.size < 6:
                    continue
                contour = np.squeeze(contour, axis=1)
                polygons[i].append(contour)

        return polygons


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

    def to_polygon(self):
        if not self.raw_mask:
            warnings.warn("Polygon can be found only for non-empty mask")
            return {}

        if not self.labels:
            warnings.warn("Polygon can be found only for non-empty labels")
            return {}

        polygons = defaultdict(list)
        for elem, label in zip(self.raw_mask, self.labels):
            elem = np.uint8(maskUtils.decode(elem))
            contours, _ = cv.findContours(elem, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if contour.size < 6:
                    continue
                contour = np.squeeze(contour, axis=1)
                polygons[label].append(contour)

        return polygons


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

    def to_annotation(self, **kwargs):
        return CoCoInstanceSegmentationAnnotation(self.identifier, self.mask, self.labels)

class OAR3DTilingSegmentationAnnotation(SegmentationAnnotation):
    def __init__(self, identifier, path_to_mask):
        super().__init__(identifier, path_to_mask, GTMaskLoader.NUMPY)

    def _load_mask(self):
        if self._mask is None:
            loader_config = self.LOADERS.get(self._mask_loader)
            data_source = self.metadata.get('segmentation_masks_source')
            if data_source is None:
                data_source = self.metadata['data_source']
            loader = BaseReader.provide(loader_config, data_source)
            mask = loader.read(self._mask_path)
            return mask

        return self._mask
