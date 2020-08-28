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

from .postprocessing_executor import PostprocessingExecutor, Postprocessor

from .filter import (
    FilterPostprocessor,

    FilterByHeightRange,
    FilterByLabels,
    FilterByMinConfidence,
    FilterEmpty,
    FilterByVisibility,
    FilterByAspectRatio
)

from .cast_to_int import CastToInt
from .clip_boxes import ClipBoxes
from .nms import NMS, SoftNMS
from .resize_prediction_boxes import ResizePredictionBoxes
from .faster_rcnn_postprocessing_resize import FRCNNPostprocessingBboxResize
from .correct_yolo_v2_boxes import CorrectYoloV2Boxes
from .resize_segmentation_mask import ResizeSegmentationMask
from .encode_segmentation_mask import EncodeSegMask
from .normalize_landmarks_points import NormalizeLandmarksPoints
from .clip_points import ClipPoints
from .extend_segmentation_mask import ExtendSegmentationMask
from .zoom_segmentation_mask import ZoomSegMask
from .crop_segmentation_mask import CropSegmentationMask, CropOrPadSegmentationMask
from .clip_segmentation_mask import ClipSegmentationMask
from .normalize_boxes import NormalizeBoxes
from .brats_postprocessing import SegmentationPredictionResample, TransformBratsPrediction
from .extract_answers_tokens import ExtractSQUADPrediction
from .translate_3d_poses import Translate3dPoses
from .normalize_recomendation import MinMaxNormalizeRecommendation, SigmoidNormalizeRecommendation
from .align_prediction_depth_map import AlignDepth
from .resize_prediction_depth_map import ResizeDepthMap
from .resize_super_resolution import ResizeSuperResolution
from .resize_style_transfer import ResizeStyleTransfer
from .crop_ground_truth_image import CropGTImage
from .resize import Resize
from .to_gray_scale_ref_image import RGB2GRAYAnnotation, BGR2GRAYAnnotation


__all__ = [
    'Postprocessor',
    'PostprocessingExecutor',

    'FilterPostprocessor',
    'FilterByHeightRange',
    'FilterByLabels',
    'FilterByMinConfidence',
    'FilterEmpty',
    'FilterByVisibility',
    'FilterByAspectRatio',

    'CastToInt',
    'ClipBoxes',
    'NMS',
    'SoftNMS',
    'ResizePredictionBoxes',
    'FRCNNPostprocessingBboxResize',
    'CorrectYoloV2Boxes',
    'NormalizeBoxes',

    'ResizeSegmentationMask',
    'EncodeSegMask',
    'ExtendSegmentationMask',
    'ZoomSegMask',
    'CropSegmentationMask',
    'ClipSegmentationMask',

    'SegmentationPredictionResample',
    'TransformBratsPrediction',

    'NormalizeLandmarksPoints',

    'ExtractSQUADPrediction',

    'Translate3dPoses',

    'SigmoidNormalizeRecommendation',
    'MinMaxNormalizeRecommendation',

    'MinMaxNormalizeRecommendation',

    'AlignDepth',
    'ResizeDepthMap',

    'ResizeSuperResolution',
    'ResizeStyleTransfer',
    'RGB2GRAYAnnotation',
    'BGR2GRAYAnnotation',

    'CropGTImage',

    'Resize'
]
