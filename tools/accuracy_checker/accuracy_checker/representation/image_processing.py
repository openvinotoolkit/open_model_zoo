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
        self._value = value
        cache = ImageProcessingAnnotation._cached_annotations
        reference = weakref.ref(self, ImageProcessingAnnotation._remove_cached_annotation)
        if value is None:
            cache.pop(reference, None)
        else:
            cache[reference] = value.nbytes
            cache.move_to_end(reference)

    @classmethod
    def _remove_cached_annotation(cls, reference):
        cls._cached_annotations.pop(reference, None)

    def release(self):
        self._set_cached_value(None)

    @classmethod
    def release_cache_if_needed(cls):
        cached_bytes = sum(cls._cached_annotations.values())
        if not cached_bytes:
            return False
        limit = _get_total_memory() * cls._cache_memory_threshold
        released = False
        while cached_bytes > limit:
            reference, size = cls._cached_annotations.popitem(last=False)
            annotation = reference()
            if annotation is not None:
                annotation.release()
            cached_bytes -= size
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
