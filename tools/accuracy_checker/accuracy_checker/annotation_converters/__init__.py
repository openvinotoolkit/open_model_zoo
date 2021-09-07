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

from .format_converter import BaseFormatConverter
from .convert import make_subset, save_annotation, analyze_dataset, DatasetConversionInfo
from .market1501 import Market1501Converter
from .veri776 import VeRi776Converter
from .mars import MARSConverter
from .pascal_voc import PascalVOCDetectionConverter
from .sample_converter import SampleConverter
from .wider import WiderFormatConverter
from .detection_opencv_storage import DetectionOpenCVStorageFormatConverter
from .lfw import LFWConverter, FaceRecognitionBinary
from .vgg_face_regression import VGGFaceRegressionConverter
from .super_resolution_converter import (
    SRConverter, SRMultiFrameConverter, MultiTargetSuperResolutionConverter, SRDirectoryBased
)
from .imagenet import ImageNetFormatConverter
from .icdar import ICDAR13RecognitionDatasetConverter, ICDAR15DetectionDatasetConverter
from .im2latex import Im2latexDatasetConverter
from .unicode_character_recognition import (
    UnicodeCharacterRecognitionDatasetConverter, KondateNakayosiRecognitionDatasetConverter
)
from .ms_coco import MSCocoDetectionConverter, MSCocoKeypointsConverter, MSCocoSingleKeypointsConverter
from .cityscapes import CityscapesConverter
from .ncf_converter import MovieLensConverter
from .brats import BratsConverter, BratsNumpyConverter
from .oar3d import OAR3DTilingConverter
from .cifar import CifarFormatConverter
from .mnist import MNISTCSVFormatConverter, MNISTFormatConverter
from .wmt import WMTConverter
from .common_semantic_segmentation import CommonSegmentationConverter
from .camvid import CamVidConverter, CamVid32DatasetConverter
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
from .squad_emb import SQUADConverterEMB
from .squad_bidaf import SQUADConverterBiDAF
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

from .image_processing import ImageProcessingConverter, ParametricImageProcessing

from .aflw2000_3d import AFLW20003DConverter
from .ade20k_dataset_converter import ADE20kConverter

from .librispeech import LibrispeechConverter
from .criteo_kaggle_dac import CriteoKaggleDACConverter
from .features_regression import FeaturesRegressionConverter, MultiOutputFeaturesRegression
from .nyu_depth import NYUDepthV2Converter
from .dna_seq import DNASequenceDatasetConverter
from .place_recognition import PlaceRecognitionDatasetConverter
from .cluttered_mnist import ClutteredMNISTConverter
from .mpii import MPIIDatasetConverter
from .mapillary_20 import Mapillary20Converter, MapillaryVistasConverter
from .antispoofing import AntispoofingDatasetConverter
from .sound_classification_converter import SoundClassificationFormatConverter
from .ade20k_image_translation import ADE20kImageTranslationConverter
from .salient_object_detection import SalientObjectDetectionConverter
from .common_object_detection import CommonDetectionConverter
from .wflw import WFLWConverter
from .see_in_the_dark import SeeInTheDarkDatasetConverter
from .conll_ner import CONLLDatasetConverter
from .background_matting import BackgroundMattingConverter
from .tacotron2_test_data_converter import TacotronDataConverter
from .noise_suppression_dataset import NoiseSuppressionDatasetConverter
from .vimeo90k_sr import Vimeo90KSuperResolutionDatasetConverter
from .lmdb import LMDBConverter
from .electricity_time_series_forecasting import ElectricityTimeSeriesForecastingConverter
from .kaldi_speech_recognition_pipeline import KaldiSpeechRecognitionDataConverter, KaldiFeatureRegressionConverter
from .yolo_labeling_converter import YOLOLabelingConverter
from .label_me_converter import LabelMeDetectionConverter, LabelMeSegmentationConverter
from .dataset_folder import DatasetFolderConverter
from .open_images_converter import OpenImagesDetectionConverter
from .calgarycampinas import KSpaceMRIConverter

__all__ = [
    'BaseFormatConverter',
    'DatasetConversionInfo',
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
    'FaceRecognitionBinary',
    'VGGFaceRegressionConverter',
    'SRConverter',
    'SRMultiFrameConverter',
    'MultiTargetSuperResolutionConverter',
    'SRDirectoryBased',
    'ICDAR13RecognitionDatasetConverter',
    'ICDAR15DetectionDatasetConverter',
    'UnicodeCharacterRecognitionDatasetConverter',
    'KondateNakayosiRecognitionDatasetConverter',
    'MSCocoKeypointsConverter',
    'MSCocoSingleKeypointsConverter',
    'MSCocoDetectionConverter',
    'CityscapesConverter',
    'Mapillary20Converter',
    'MapillaryVistasConverter',
    'MovieLensConverter',
    'BratsConverter',
    'BratsNumpyConverter',
    'OAR3DTilingConverter',
    'CifarFormatConverter',
    'MNISTCSVFormatConverter',
    'MNISTFormatConverter',
    'WMTConverter',
    'CommonSegmentationConverter',
    'CamVidConverter',
    'CamVid32DatasetConverter',
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
    'SQUADConverterEMB',
    'SQUADConverterBiDAF',
    'XNLIDatasetConverter',
    'BertXNLITFRecordConverter',
    'IMDBConverter',
    'MRPCConverter',
    'CoLAConverter',
    'CmuPanopticKeypointsConverter',
    'ActionRecognitionConverter',
    'MSASLContiniousConverter',
    'ReDWebDatasetConverter',
    'NYUDepthV2Converter',
    'FashionMnistConverter',
    'InpaintingConverter',
    'mrlEyes_2018_01_Converter',
    'StyleTransferConverter',
    'Wikitext2RawConverter',
    'ImageProcessingConverter',
    'AFLW20003DConverter',
    'ADE20kConverter',
    'LibrispeechConverter',
    'CriteoKaggleDACConverter',
    'FeaturesRegressionConverter',
    'MultiOutputFeaturesRegression',
    'Im2latexDatasetConverter',
    'DNASequenceDatasetConverter',
    'PlaceRecognitionDatasetConverter',
    'ClutteredMNISTConverter',
    'MPIIDatasetConverter',
    'AntispoofingDatasetConverter',
    'SoundClassificationFormatConverter',
    'ADE20kImageTranslationConverter',
    'SalientObjectDetectionConverter',
    'CommonDetectionConverter',
    'WFLWConverter',
    'SeeInTheDarkDatasetConverter',
    'CONLLDatasetConverter',
    'BackgroundMattingConverter',
    'TacotronDataConverter',
    'NoiseSuppressionDatasetConverter',
    'Vimeo90KSuperResolutionDatasetConverter',
    'LMDBConverter',
    'ElectricityTimeSeriesForecastingConverter',
    'KaldiSpeechRecognitionDataConverter',
    'KaldiFeatureRegressionConverter',
    'ParametricImageProcessing',
    'YOLOLabelingConverter',
    'LabelMeDetectionConverter',
    'LabelMeSegmentationConverter',
    'DatasetFolderConverter',
    'OpenImagesDetectionConverter',
    'KSpaceMRIConverter'
]
