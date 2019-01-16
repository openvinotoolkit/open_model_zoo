"""
Copyright (c) 2018 Intel Corporation

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
from .classification_representation import Classification, ClassificationAnnotation, ClassificationPrediction
from .detection_representation import Detection, DetectionAnnotation, DetectionPrediction
from .reid_representation import (ReIdentificationPrediction, ReIdentificationAnnotation,
                                  ReIdentificationClassificationAnnotation)
from .segmentation_representation import SegmentationAnnotation, SegmentationPrediction, SegmentationRepresentation
from .character_recognition_representation import (CharacterRecognition, CharacterRecognitionAnnotation,
                                                   CharacterRecognitionPrediction)
from .representaton_container import ContainerRepresentation, ContainerPrediction, ContainerAnnotation
from .regression_representation import (RegressionAnnotation, RegressionPrediction, FacialLandmarksAnnotation,
                                        FacialLandmarksPrediction)
from .multilabel_recognition import MultilabelRecognitionAnnotation, MultilabelRecognitionPrediction
from .super_resolution_representation import SuperResolutionAnnotation, SuperResolutionPrediction

__all__ = [
    'BaseRepresentation',
    'Detection', 'Classification',
    'DetectionAnnotation', 'DetectionPrediction',
    'ClassificationAnnotation', 'ClassificationPrediction',
    'ReIdentificationPrediction', 'ReIdentificationAnnotation', 'ReIdentificationClassificationAnnotation',
    'SegmentationRepresentation', 'SegmentationAnnotation', 'SegmentationPrediction',
    'CharacterRecognition', 'CharacterRecognitionPrediction', 'CharacterRecognitionAnnotation',
    'ContainerRepresentation', 'ContainerAnnotation', 'ContainerPrediction',
    'RegressionAnnotation', 'RegressionPrediction', 'FacialLandmarksAnnotation', 'FacialLandmarksPrediction',
    'MultilabelRecognitionAnnotation', 'MultilabelRecognitionPrediction',
    'SuperResolutionAnnotation', 'SuperResolutionPrediction'
]
