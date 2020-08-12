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

from .adapter import Adapter, AdapterField, create_adapter

from .action_recognition import ActionDetection
from .text_detection import (
    TextDetectionAdapter,
    TextProposalsDetectionAdapter,
    EASTTextDetectionAdapter
)

from .text_recognition import (
    BeamSearchDecoder,
    CTCGreedySearchDecoder,
    LPRAdapter
)

from .image_processing import (
    ImageProcessingAdapter, SuperResolutionAdapter, MultiSuperResolutionAdapter, SuperResolutionYUV
)
from .attributes_recognition import (
    HeadPoseEstimatorAdapter,
    VehicleAttributesRecognitionAdapter,
    PersonAttributesAdapter,
    AgeGenderAdapter,
    LandmarksRegressionAdapter,
    GazeEstimationAdapter,
    PRNetAdapter
)

from .reidentification import ReidAdapter
from .detection import (
    TFObjectDetectionAPIAdapter,
    MTCNNPAdapter,
    RetinaNetAdapter,
    ClassAgnosticDetectionAdapter,
    FaceBoxesAdapter,
    FaceDetectionAdapter,
    FaceDetectionRefinementAdapter
)
from .detection_person_vehicle import (
    PersonVehicleDetectionAdapter,
    PersonVehicleDetectionRefinementAdapter
)
from .detection_head import HeadDetectionAdapter
from .ssd import SSDAdapter, PyTorchSSDDecoder, FacePersonAdapter, SSDAdapterMxNet, SSDONNXAdapter
from .retinaface import RetinaFaceAdapter
from .yolo import TinyYOLOv1Adapter, YoloV2Adapter, YoloV3Adapter
from .classification import ClassificationAdapter
from .segmentation import SegmentationAdapter, BrainTumorSegmentationAdapter
from .pose_estimation import HumanPoseAdapter
from .pose_estimation_3d import HumanPose3dAdapter

from .hit_ratio import HitRatioAdapter

from .mask_rcnn import MaskRCNNAdapter
from .mask_rcnn_with_text import MaskRCNNWithTextAdapter
from .yolact import YolactAdapter

from .nlp import MachineTranslationAdapter, QuestionAnsweringAdapter

from .centernet import CTDETAdapter

from .mono_depth import MonoDepthAdapter

from .image_inpainting import ImageInpaintingAdapter
from .style_transfer import StyleTransferAdapter

from .attribute_classification import AttributeClassificationAdapter
from .audio_recognition import CTCBeamSearchDecoder, CTCGreedyDecoder

from .regression import RegressionAdapter
from .mixed_adapter import MixedAdapter
from .face_recognition_quality_assessment import QualityAssessmentAdapter
from .dummy_adapters import GVADetectionAdapter, XML2DetectionAdapter, GVAClassificationAdapter

__all__ = [
    'Adapter',
    'AdapterField',
    'create_adapter',

    'XML2DetectionAdapter',

    'ClassificationAdapter',

    'TFObjectDetectionAPIAdapter',
    'MTCNNPAdapter',
    'CTDETAdapter',
    'RetinaNetAdapter',
    'ClassAgnosticDetectionAdapter',
    'RetinaFaceAdapter',
    'FaceBoxesAdapter',
    'FaceDetectionAdapter',
    'FaceDetectionRefinementAdapter',
    'PersonVehicleDetectionAdapter',
    'PersonVehicleDetectionRefinementAdapter',
    'HeadDetectionAdapter',

    'SegmentationAdapter',
    'BrainTumorSegmentationAdapter',

    'ReidAdapter',

    'ImageProcessingAdapter',
    'SuperResolutionAdapter',
    'MultiSuperResolutionAdapter',
    'SuperResolutionYUV',

    'HeadPoseEstimatorAdapter',
    'VehicleAttributesRecognitionAdapter',
    'PersonAttributesAdapter',
    'AgeGenderAdapter',
    'LandmarksRegressionAdapter',
    'GazeEstimationAdapter',
    'PRNetAdapter',

    'TextDetectionAdapter',
    'TextProposalsDetectionAdapter',
    'EASTTextDetectionAdapter',

    'BeamSearchDecoder',
    'LPRAdapter',
    'CTCGreedySearchDecoder',

    'HumanPoseAdapter',
    'HumanPose3dAdapter',

    'ActionDetection',

    'HitRatioAdapter',

    'MaskRCNNAdapter',
    'MaskRCNNWithTextAdapter',
    'YolactAdapter',

    'MachineTranslationAdapter',
    'QuestionAnsweringAdapter',

    'MonoDepthAdapter',

    'ImageInpaintingAdapter',
    'StyleTransferAdapter',

    'AttributeClassificationAdapter',

    'RegressionAdapter',
    'MixedAdapter',

    'CTCBeamSearchDecoder',
    'CTCGreedyDecoder',

    'QualityAssessmentAdapter',

    'GVADetectionAdapter',
    'GVAClassificationAdapter',

]
