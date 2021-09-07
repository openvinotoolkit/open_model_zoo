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

from .adapter import Adapter, AdapterField, create_adapter

from .action_recognition import ActionDetection
from .text_detection import (
    TextDetectionAdapter,
    TextProposalsDetectionAdapter,
    EASTTextDetectionAdapter,
    CRAFTTextDetectionAdapter
)

from .text_recognition import (
    BeamSearchDecoder,
    CTCGreedySearchDecoder,
    LPRAdapter,
    AttentionOCRAdapter,
    SimpleDecoder,
    PDPDTextRecognition
)

from .image_processing import (
    ImageProcessingAdapter, SuperResolutionAdapter, MultiSuperResolutionAdapter, SuperResolutionYUV, TrimapAdapter
)
from .attributes_recognition import (
    HeadPoseEstimatorAdapter,
    VehicleAttributesRecognitionAdapter,
    PersonAttributesAdapter,
    AgeGenderAdapter,
    AgeRecognitionAdapter,
    LandmarksRegressionAdapter,
    GazeEstimationAdapter,
    PRNetAdapter
)

from .reidentification import ReidAdapter
from .detection import (
    TFObjectDetectionAPIAdapter,
    MTCNNPAdapter,
    ClassAgnosticDetectionAdapter,
    FaceBoxesAdapter,
    FaceDetectionAdapter,
    FaceDetectionRefinementAdapter,
    FasterRCNNONNX,
    TwoStageDetector,
    DETRAdapter,
    UltraLightweightFaceDetectionAdapter
)
from .detection_person_vehicle import (
    PersonVehicleDetectionAdapter,
    PersonVehicleDetectionRefinementAdapter
)
from .detection_head import HeadDetectionAdapter
from .ssd import SSDAdapter, PyTorchSSDDecoder, FacePersonAdapter, SSDAdapterMxNet, SSDONNXAdapter
from .retinaface import RetinaFaceAdapter, RetinaFacePyTorchAdapter
from .retinanet import RetinaNetAdapter, MultiOutRetinaNet, RetinaNetTF2
from .yolo import TinyYOLOv1Adapter, YoloV2Adapter, YoloV3Adapter, YoloV3ONNX, YoloV3TF2, YoloV5Adapter
from .classification import ClassificationAdapter
from .segmentation import (
    SegmentationAdapter, BrainTumorSegmentationAdapter, DUCSegmentationAdapter, BackgroundMattingAdapter
)
from .pose_estimation import HumanPoseAdapter, SingleHumanPoseAdapter, StackedHourGlassNetworkAdapter
from .pose_estimation_openpose import OpenPoseAdapter
from .pose_estimation_associative_embedding import AssociativeEmbeddingAdapter
from .pose_estimation_hrnet import HumanPoseHRNetAdapter

from .pose_estimation_3d import HumanPose3dAdapter

from .hit_ratio import HitRatioAdapter

from .mask_rcnn import MaskRCNNAdapter
from .mask_rcnn_with_text import MaskRCNNWithTextAdapter
from .yolact import YolactAdapter

from .nlp import (
    MachineTranslationAdapter, QuestionAnsweringAdapter, QuestionAnsweringBiDAFAdapter,
    BertTextClassification, BERTNamedEntityRecognition
)

from .centernet import CTDETAdapter

from .mono_depth import MonoDepthAdapter

from .image_inpainting import ImageInpaintingAdapter
from .style_transfer import StyleTransferAdapter

from .attribute_classification import AttributeClassificationAdapter
from .audio_recognition import (
    CTCBeamSearchDecoder,
    CTCGreedyDecoder,
    CTCBeamSearchDecoderWithLm,
    FastCTCBeamSearchDecoderWithLm
)
from .kaldi_asr_decoder import KaldiLatGenDecoder
from .regression import RegressionAdapter, MultiOutputRegression
from .mixed_adapter import MixedAdapter
from .face_recognition_quality_assessment import QualityAssessmentAdapter
from .dna_seq_recognition import DNASeqRecognition
from .optical_flow import PWCNetAdapter
from .salient_objects_detection import SalientObjectDetection
from .noise_suppression import NoiseSuppressionAdapter
from .dummy_adapters import GVADetectionAdapter, XML2DetectionAdapter, GVAClassificationAdapter

from .time_series import QuantilesPredictorAdapter

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
    'RetinaNetTF2',
    'ClassAgnosticDetectionAdapter',
    'RetinaFaceAdapter',
    'RetinaFacePyTorchAdapter',
    'FaceBoxesAdapter',
    'FaceDetectionAdapter',
    'FaceDetectionRefinementAdapter',
    'PersonVehicleDetectionAdapter',
    'PersonVehicleDetectionRefinementAdapter',
    'HeadDetectionAdapter',
    'FasterRCNNONNX',
    'TwoStageDetector',
    'DETRAdapter',
    'UltraLightweightFaceDetectionAdapter',

    'TinyYOLOv1Adapter',
    'YoloV2Adapter',
    'YoloV3Adapter',
    'YoloV3ONNX',
    'YoloV3TF2',
    'YoloV5Adapter',

    'SSDAdapter',
    'SSDAdapterMxNet',
    'SSDONNXAdapter',
    'PyTorchSSDDecoder',
    'FacePersonAdapter',

    'RetinaNetAdapter',
    'MultiOutRetinaNet',

    'SegmentationAdapter',
    'BrainTumorSegmentationAdapter',
    'DUCSegmentationAdapter',
    'SalientObjectDetection',
    'BackgroundMattingAdapter',

    'ReidAdapter',

    'ImageProcessingAdapter',
    'SuperResolutionAdapter',
    'MultiSuperResolutionAdapter',
    'SuperResolutionYUV',
    'TrimapAdapter',

    'HeadPoseEstimatorAdapter',
    'VehicleAttributesRecognitionAdapter',
    'PersonAttributesAdapter',
    'AgeGenderAdapter',
    'AgeRecognitionAdapter',
    'LandmarksRegressionAdapter',
    'GazeEstimationAdapter',
    'PRNetAdapter',

    'TextDetectionAdapter',
    'TextProposalsDetectionAdapter',
    'EASTTextDetectionAdapter',
    'CRAFTTextDetectionAdapter',

    'BeamSearchDecoder',
    'LPRAdapter',
    'CTCGreedySearchDecoder',
    'AttentionOCRAdapter',
    'SimpleDecoder',
    'PDPDTextRecognition',

    'AssociativeEmbeddingAdapter',
    'HumanPoseAdapter',
    'HumanPose3dAdapter',
    'HumanPoseHRNetAdapter',
    'OpenPoseAdapter',
    'SingleHumanPoseAdapter',
    'StackedHourGlassNetworkAdapter',

    'ActionDetection',

    'HitRatioAdapter',

    'MaskRCNNAdapter',
    'MaskRCNNWithTextAdapter',
    'YolactAdapter',

    'MachineTranslationAdapter',
    'QuestionAnsweringAdapter',
    'QuestionAnsweringBiDAFAdapter',
    'BERTNamedEntityRecognition',
    'BertTextClassification',

    'MonoDepthAdapter',

    'ImageInpaintingAdapter',
    'StyleTransferAdapter',

    'AttributeClassificationAdapter',

    'RegressionAdapter',
    'MultiOutputRegression',
    'MixedAdapter',

    'CTCBeamSearchDecoder',
    'CTCGreedyDecoder',
    'CTCBeamSearchDecoderWithLm',
    'FastCTCBeamSearchDecoderWithLm',
    'KaldiLatGenDecoder',

    'QualityAssessmentAdapter',

    'DNASeqRecognition',

    'PWCNetAdapter',

    'NoiseSuppressionAdapter',

    'GVADetectionAdapter',
    'GVAClassificationAdapter',

    'QuantilesPredictorAdapter'
]
