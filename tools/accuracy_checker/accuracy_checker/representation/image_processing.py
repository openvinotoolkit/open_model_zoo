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

from collections import OrderedDict
from enum import Enum
import ctypes
import os
import weakref

import numpy as np

from .base_representation import BaseRepresentation
from ..data_readers import BaseReader


def _get_total_memory():
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return status.ullTotalPhys

    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')


class GTLoader(Enum):
    PILLOW = 0
    OPENCV = 1
    DICOM = 2
    RAWPY = 3
    SKIMAGE = 4
    PILLOW_RGB = 5
    NUMPY = 6


class ImageProcessingRepresentation(BaseRepresentation):
    pass


class ImageProcessingAnnotation(ImageProcessingRepresentation):
    _cached_annotations = OrderedDict()
    _cached_bytes = 0
    _cache_memory_threshold = 0.5

    LOADERS = {
        GTLoader.PILLOW: 'pillow_imread',
        GTLoader.OPENCV: 'opencv_imread',
        GTLoader.DICOM: 'dicom_reader',
        GTLoader.RAWPY: 'rawpy',
        GTLoader.SKIMAGE: 'skimage_imread',
        GTLoader.PILLOW_RGB: 'pillow_imread',
        GTLoader.NUMPY: 'numpy_reader'
    }

    def __init__(self, identifier, path_to_gt, gt_loader=GTLoader.PILLOW):
        """
        Args:
            identifier: object identifier (e.g. image name).
            path_to_gt: path where gt image should be loaded from. The path is relative to data source.
            gt_loader: back-end, used to load segmentation masks.
        """

        super().__init__(identifier)
        self._image_path = path_to_gt
        self._gt_loader = self.LOADERS.get(gt_loader)
        self._pillow_to_rgb = gt_loader == GTLoader.PILLOW_RGB
        self._value = None

    @property
    def value(self):
        if self._value is None:
            data_source = self.metadata.get('additional_data_source')
            if not data_source:
                data_source = self.metadata['data_source']
            loader = BaseReader.provide(self._gt_loader, data_source)
            if self._gt_loader == self.LOADERS[GTLoader.PILLOW]:
                loader.convert_to_rgb = self._pillow_to_rgb if hasattr(self, '_pillow_to_rgb') else False
            gt = loader.read(self._image_path)
            self._value = gt.astype(np.uint8) if self._gt_loader not in ['dicom_reader', 'rawpy',
                                                                         'numpy_reader'] else gt
        self._set_cached_value(self._value)
        return self._value

    @value.setter
    def value(self, value):
        self._set_cached_value(value)

    def _set_cached_value(self, value):
        # A view would keep its whole base array alive while only its own nbytes are counted.
        if isinstance(value, np.ndarray) and value.base is not None:
            value = value.copy()
        self._value = value
        # Shared cache state lives on the base class, so subclasses must not shadow it via cls/self.
        cache = ImageProcessingAnnotation._cached_annotations
        reference = weakref.ref(self, ImageProcessingAnnotation._remove_cached_annotation)
        ImageProcessingAnnotation._cached_bytes -= cache.pop(reference, 0)
        if value is not None:
            cache[reference] = value.nbytes
            ImageProcessingAnnotation._cached_bytes += value.nbytes

    @staticmethod
    def _remove_cached_annotation(reference):
        ImageProcessingAnnotation._cached_bytes -= ImageProcessingAnnotation._cached_annotations.pop(reference, 0)

    def release(self):
        self._set_cached_value(None)

    @staticmethod
    def release_cache_if_needed():
        if not ImageProcessingAnnotation._cached_bytes:
            return False
        limit = _get_total_memory() * ImageProcessingAnnotation._cache_memory_threshold
        released = False
        while ImageProcessingAnnotation._cached_bytes > limit:
            reference, size = ImageProcessingAnnotation._cached_annotations.popitem(last=False)
            ImageProcessingAnnotation._cached_bytes -= size
            annotation = reference()
            if annotation is not None:
                # The entry is already popped, so release() does not subtract its size again.
                annotation.release()
            released = True
        return released


class ImageProcessingPrediction(ImageProcessingRepresentation):
    def __init__(self, identifiers, prediction):
        """
        Args:
            identifiers: object identifier (e.g. image name).
            prediction: array with shape (height, width) contained result image.
        """

        super().__init__(identifiers)
        self.value = prediction
