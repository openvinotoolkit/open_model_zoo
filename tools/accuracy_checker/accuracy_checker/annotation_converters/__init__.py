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
from .camvid import CamVidConverter
from .lpr import LPRConverter
from .image_retrieval import ImageRetrievalConverter
from .cvat_object_detection import CVATObjectDetectionConverter
from .cvat_attributes_recognition import CVATAttributesRecognitionConverter
from .cvat_age_gender_recognition import CVATAgeGenderRecognitionConverter
from .cvat_facial_landmarks import CVATFacialLandmarksRecognitionConverter
from .cvat_text_recognition import CVATTextRecognitionConverter
from .cvat_multilabel_recognition import CVATMultilabelAttributesRecognitionConverter
from .cvat_human_pose import CVATPoseEstimationConverter
from .cvat_person_detection_action_recognition import CVATPersonDetectionActionRecognitionConverter

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
    'CamVidConverter',
    'LPRConverter',
    'ImageRetrievalConverter',
    'CVATObjectDetectionConverter',
    'CVATAttributesRecognitionConverter',
    'CVATAgeGenderRecognitionConverter',
    'CVATFacialLandmarksRecognitionConverter',
    'CVATTextRecognitionConverter',
    'CVATMultilabelAttributesRecognitionConverter',
    'CVATPoseEstimationConverter',
    'CVATPersonDetectionActionRecognitionConverter'
]
