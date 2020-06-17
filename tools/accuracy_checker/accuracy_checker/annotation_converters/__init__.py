"""
Copyright (c) 2019-2020 Intel Corporation

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
from .veri776 import VeRi776Converter
from .mars import MARSConverter
from .pascal_voc import PascalVOCDetectionConverter
from .sample_converter import SampleConverter
from .wider import WiderFormatConverter
from .detection_opencv_storage import DetectionOpenCVStorageFormatConverter
from .lfw import LFWConverter
from .vgg_face_regression import VGGFaceRegressionConverter
from .super_resolution_converter import SRConverter, SRMultiFrameConverter, MultiTargetSuperResolutionConverter
from .imagenet import ImageNetFormatConverter
from .icdar import ICDAR13RecognitionDatasetConverter, ICDAR15DetectionDatasetConverter
from .kondate_nakayosi import KondateNakayosiRecognitionDatasetConverter
from .ms_coco import MSCocoDetectionConverter, MSCocoKeypointsConverter, MSCocoSingleKeypointsConverter
from .cityscapes import CityscapesConverter
from .ncf_converter import MovieLensConverter
from .brats import BratsConverter, BratsNumpyConverter
from .oar3d import OAR3DTilingConverter
from .cifar import CifarFormatConverter
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
from .mrlEyes_2018_01 import mrlEyes_2018_01_Converter
from .squad import SQUADConverter
from .text_classification import (
    XNLIDatasetConverter,
    BertXNLITFRecordConverter,
    IMDBConverter,
    MRPCConverter,
    CoLAConverter
)
from .cmu_panoptic import CmuPanopticKeypointsConverter
from .action_recognition import ActionRecognitionConverter
from .ms_asl_continuous import MSASLContiniousConverter

from .monocular_depth_perception import ReDWebDatasetConverter

from .fashion_mnist import FashionMnistConverter
from .inpainting import InpaintingConverter
from .style_transfer import StyleTransferConverter
from .wikitext2raw import Wikitext2RawConverter

from  .image_processing import ImageProcessingConverter

from .aflw2000_3d import AFLW20003DConverter

__all__ = [
    'BaseFormatConverter',
    'make_subset',
    'save_annotation',
    'analyze_dataset',

    'ImageNetFormatConverter',
    'Market1501Converter',
    'VeRi776Converter',
    'SampleConverter',
    'PascalVOCDetectionConverter',
    'WiderFormatConverter',
    'MARSConverter',
    'DetectionOpenCVStorageFormatConverter',
    'LFWConverter',
    'VGGFaceRegressionConverter',
    'SRConverter',
    'SRMultiFrameConverter',
    'MultiTargetSuperResolutionConverter',
    'ICDAR13RecognitionDatasetConverter',
    'ICDAR15DetectionDatasetConverter',
    'KondateNakayosiRecognitionDatasetConverter',
    'MSCocoKeypointsConverter',
    'MSCocoSingleKeypointsConverter',
    'MSCocoDetectionConverter',
    'CityscapesConverter',
    'MovieLensConverter',
    'BratsConverter',
    'BratsNumpyConverter',
    'OAR3DTilingConverter',
    'CifarFormatConverter',
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
    'CVATPersonDetectionActionRecognitionConverter',
    'SQUADConverter',
    'XNLIDatasetConverter',
    'BertXNLITFRecordConverter',
    'IMDBConverter',
    'MRPCConverter',
    'CoLAConverter',
    'CmuPanopticKeypointsConverter',
    'ActionRecognitionConverter',
    'MSASLContiniousConverter',
    'ReDWebDatasetConverter',
    'FashionMnistConverter',
    'InpaintingConverter',
    'mrlEyes_2018_01_Converter',
    'StyleTransferConverter',
    'Wikitext2RawConverter',
    'ImageProcessingConverter',
    'AFLW20003DConverter'
]
