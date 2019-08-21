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

from .format_converter import BaseFormatConverter
from .convert import make_subset, save_annotation, analyze_dataset
from .market1501 import Market1501Converter
from .mars import MARSConverter
from .pascal_voc import PascalVOCDetectionConverter
from .sample_converter import SampleConverter
from .wider import WiderFormatConverter
from .detection_opencv_storage import DetectionOpenCVStorageFormatConverter
from .lfw import LFWConverter
from .vgg_face_regression import VGGFaceRegressionConverter
from .super_resolution_converter import SRConverter
from .imagenet import ImageNetFormatConverter
from .icdar import ICDAR13RecognitionDatasetConverter, ICDAR15DetectionDatasetConverter
from .ms_coco import MSCocoDetectionConverter, MSCocoKeypointsConverter
from .cityscapes import CityscapesConverter
from .ncf_converter import MovieLensConverter
from .brats import BratsConverter, BratsNumpyConverter
from .cifar10 import Cifar10FormatConverter
from .mnist import MNISTCSVFormatConverter
from .wmt import WMTConverter
from .common_semantic_segmentation import CommonSegmentationConverter
from .cvat_object_detection import CVATObjectDetectionConverter
from .cvat_attributes_recognition import CVATAttributesRecognitionConverter
from .cvat_age_gender_recognition import CVATAgeGenderRecognitionConverter

__all__ = [
    'BaseFormatConverter',
    'make_subset',
    'save_annotation',
    'analyze_dataset',

    'ImageNetFormatConverter',
    'Market1501Converter',
    'SampleConverter',
    'PascalVOCDetectionConverter',
    'WiderFormatConverter',
    'MARSConverter',
    'DetectionOpenCVStorageFormatConverter',
    'LFWConverter',
    'VGGFaceRegressionConverter',
    'SRConverter',
    'ICDAR13RecognitionDatasetConverter',
    'ICDAR15DetectionDatasetConverter',
    'MSCocoKeypointsConverter',
    'MSCocoDetectionConverter',
    'CityscapesConverter',
    'MovieLensConverter',
    'BratsConverter',
    'BratsNumpyConverter',
    'Cifar10FormatConverter',
    'MNISTCSVFormatConverter',
    'WMTConverter',
    'CommonSegmentationConverter',
    'CVATObjectDetectionConverter',
    'CVATAttributesRecognitionConverter',
    'CVATAgeGenderRecognitionConverter'
]
