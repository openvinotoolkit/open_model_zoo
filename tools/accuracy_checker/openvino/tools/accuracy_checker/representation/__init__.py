"""
Copyright (c) 2018-2022 Intel Corporation

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

from .base_representation import BaseRepresentation
from .classification_representation import (
    Classification,
    ClassificationAnnotation,
    ClassificationPrediction,
    ArgMaxClassificationPrediction,
    SequenceClassificationAnnotation,
    SequenceClassificationPrediction
)
from .detection_representation import (
    Detection,
    DetectionAnnotation,
    DetectionPrediction,
    AttributeDetectionAnnotation,
    AttributeDetectionPrediction,
    ActionDetectionAnnotation,
    ActionDetectionPrediction
)
from .reid_representation import (
    ReIdentificationAnnotation,
    ReIdentificationClassificationAnnotation,
    PlaceRecognitionAnnotation,
    ReIdentificationPrediction
)
from .segmentation_representation import (
    SegmentationRepresentation,
    SegmentationAnnotation,
    SegmentationPrediction,
    BrainTumorSegmentationAnnotation,
    BrainTumorSegmentationPrediction,
    CoCoInstanceSegmentationAnnotation,
    CoCoInstanceSegmentationPrediction,
    OAR3DTilingSegmentationAnnotation,
    SalientRegionAnnotation,
    SalientRegionPrediction,
    BackgroundMattingAnnotation,
    BackgroundMattingPrediction,
    AnomalySegmentationAnnotation,
    AnomalySegmentationPrediction
)
from .character_recognition_representation import (
    CharacterRecognition,
    CharacterRecognitionAnnotation,
    CharacterRecognitionPrediction
)
from .representaton_container import ContainerRepresentation, ContainerAnnotation, ContainerPrediction
from .regression_representation import (
    RegressionAnnotation,
    RegressionPrediction,
    FacialLandmarksAnnotation,
    FacialLandmarksPrediction,
    FacialLandmarks3DAnnotation,
    FacialLandmarks3DPrediction,
    GazeVectorAnnotation,
    GazeVectorPrediction,
    FeaturesRegressionAnnotation,
    NiftiRegressionAnnotation,
    HandLandmarksAnnotation,
    HandLandmarksPrediction,
)
from .facial_landmarks_heatmap_representation import FacialLandmarksHeatMapAnnotation, FacialLandmarksHeatMapPrediction
from .multilabel_recognition import MultiLabelRecognitionAnnotation, MultiLabelRecognitionPrediction
from .super_resolution_representation import SuperResolutionAnnotation, SuperResolutionPrediction
from .text_detection_representation import TextDetectionAnnotation, TextDetectionPrediction
from .pose_estimation_representation import PoseEstimationAnnotation, PoseEstimationPrediction
from .pose_estimation_3d_representation import PoseEstimation3dAnnotation, PoseEstimation3dPrediction
from .hit_ratio_representation import HitRatio, HitRatioAnnotation, HitRatioPrediction
from .nlp_representation import (
    MachineTranslationAnnotation,
    MachineTranslationPrediction,
    QuestionAnsweringAnnotation,
    QuestionAnsweringPrediction,
    QuestionAnsweringEmbeddingAnnotation,
    QuestionAnsweringEmbeddingPrediction,
    TextClassificationAnnotation,
    LanguageModelingAnnotation,
    LanguageModelingPrediction,
    QuestionAnsweringBiDAFAnnotation,
    BERTNamedEntityRecognitionAnnotation,
    SentenceSimilarityAnnotation
)
from .image_inpainting import ImageInpaintingAnnotation, ImageInpaintingPrediction
from .style_transfer import StyleTransferAnnotation, StyleTransferPrediction

from .depth_estimation import DepthEstimationAnnotation, DepthEstimationPrediction
from .image_processing import ImageProcessingAnnotation, ImageProcessingPrediction
from .quality_assessment import QualityAssessmentAnnotation, QualityAssessmentPrediction
from .dna_sequence import DNASequenceAnnotation, DNASequencePrediction

from .raw_representation import RawTensorAnnotation, RawTensorPrediction

from .optical_flow import OpticalFlowAnnotation, OpticalFlowPrediction

from .noise_suppression import NoiseSuppressionAnnotation, NoiseSuppressionPrediction

from .time_series_representation import (
    TimeSeriesForecastingAnnotation,
    TimeSeriesForecastingQuantilesPrediction
)

__all__ = [
    'BaseRepresentation',

    'Classification',
    'ClassificationAnnotation',
    'ClassificationPrediction',
    'ArgMaxClassificationPrediction',
    'SequenceClassificationAnnotation',
    'SequenceClassificationPrediction',

    'Detection',
    'DetectionAnnotation',
    'DetectionPrediction',

    'AttributeDetectionAnnotation',
    'AttributeDetectionPrediction',
    'ActionDetectionAnnotation',
    'ActionDetectionPrediction',

    'ReIdentificationAnnotation',
    'ReIdentificationClassificationAnnotation',
    'PlaceRecognitionAnnotation',
    'ReIdentificationPrediction',

    'SegmentationRepresentation',
    'SegmentationAnnotation',
    'SegmentationPrediction',

    'SalientRegionAnnotation',
    'SalientRegionPrediction',

    'BackgroundMattingAnnotation',
    'BackgroundMattingPrediction',

    'AnomalySegmentationAnnotation',
    'AnomalySegmentationPrediction',

    'BrainTumorSegmentationAnnotation',
    'BrainTumorSegmentationPrediction',
    'OAR3DTilingSegmentationAnnotation',

    'CoCoInstanceSegmentationAnnotation',
    'CoCoInstanceSegmentationPrediction',

    'CharacterRecognition',
    'CharacterRecognitionAnnotation',
    'CharacterRecognitionPrediction',

    'ContainerRepresentation',
    'ContainerAnnotation',
    'ContainerPrediction',

    'RegressionAnnotation',
    'RegressionPrediction',
    'FacialLandmarksHeatMapAnnotation',
    'FacialLandmarksHeatMapPrediction',
    'FacialLandmarksAnnotation',
    'FacialLandmarksPrediction',
    'FacialLandmarks3DAnnotation',
    'FacialLandmarks3DPrediction',
    'GazeVectorAnnotation',
    'GazeVectorPrediction',
    'FeaturesRegressionAnnotation',

    'MultiLabelRecognitionAnnotation',
    'MultiLabelRecognitionPrediction',

    'SuperResolutionAnnotation',
    'SuperResolutionPrediction',
    'ImageInpaintingAnnotation',
    'ImageInpaintingPrediction',
    'ImageProcessingAnnotation',
    'ImageProcessingPrediction',
    'StyleTransferAnnotation',
    'StyleTransferPrediction',

    'TextDetectionAnnotation',
    'TextDetectionPrediction',

    'PoseEstimationAnnotation',
    'PoseEstimationPrediction',
    'PoseEstimation3dAnnotation',
    'PoseEstimation3dPrediction',

    'HitRatio',
    'HitRatioAnnotation',
    'HitRatioPrediction',

    'MachineTranslationAnnotation',
    'MachineTranslationPrediction',
    'QuestionAnsweringAnnotation',
    'QuestionAnsweringPrediction',
    'QuestionAnsweringEmbeddingAnnotation',
    'QuestionAnsweringEmbeddingPrediction',
    'QuestionAnsweringBiDAFAnnotation',
    'TextClassificationAnnotation',
    'LanguageModelingAnnotation',
    'LanguageModelingPrediction',
    'BERTNamedEntityRecognitionAnnotation',
    'SentenceSimilarityAnnotation',

    'DepthEstimationAnnotation',
    'DepthEstimationPrediction',

    'QualityAssessmentAnnotation',
    'QualityAssessmentPrediction',

    'DNASequenceAnnotation',
    'DNASequencePrediction',

    'RawTensorAnnotation',
    'RawTensorPrediction',

    'OpticalFlowAnnotation',
    'OpticalFlowPrediction',

    'NiftiRegressionAnnotation',

    'NoiseSuppressionAnnotation',
    'NoiseSuppressionPrediction',

    'TimeSeriesForecastingAnnotation',
    'TimeSeriesForecastingQuantilesPrediction',

    'HandLandmarksAnnotation',
    'HandLandmarksPrediction'
]
