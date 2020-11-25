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

from .preprocessing_executor import PreprocessingExecutor
from .preprocessor import Preprocessor
from .color_space_conversion import (
    BgrToRgb, RgbToBgr,
    BgrToGray, RgbToGray,
    TfConvertImageDType,
    SelectInputChannel,
    BGR2YUVConverter, RGB2YUVConverter,
    BGRtoNV12Converter, RGBtoNV12Converter,
    NV12toBGRConverter, NV12toRGBConverter,
    RGB2YCrCbConverter, BGR2YCrCbConverter,
    BGRToLAB, RGBToLAB
)
from .audio_preprocessing import (
    ResampleAudio,
    ClipAudio,
    NormalizeAudio,
    HanningWindow,
    AudioSpectrogram,
    TriangleFiltering,
    DCT,
    ClipCepstrum,
    TrimmingAudio,
    SamplesToFloat32,
    AudioToMelSpectrogram
)

from .normalization import Normalize, Normalize3d
from .geometric_transformations import (
    GeometricOperationMetadata,
    Flip,
    PointAligner,
    Tiling,
    ImagePyramid,
    FaceDetectionImagePyramid,
    WarpAffine
)
from .crop import (
    Crop, CropRect, ExtendAroundRect, Crop3D, TransformedCropWithAutoScale,
    CandidateCrop, CropOrPad, CropWithPadSize, CornerCrop, ObjectCropWithScale
)
from .resize import Resize, AutoResize
from .nlp_preprocessors import DecodeByVocabulary, PadWithEOS
from .centernet_preprocessing import CenterNetAffineTransformation
from .brats_preprocessing import Resize3D, NormalizeBrats, CropBraTS, SwapModalitiesBrats
from .inpainting_preprocessor import FreeFormMask, RectMask, CustomMask
from .one_hot_encoding import OneHotEncoding

__all__ = [
    'PreprocessingExecutor',

    'Preprocessor',
    'GeometricOperationMetadata',

    'ResampleAudio',
    'ClipAudio',
    'NormalizeAudio',
    'HanningWindow',
    'AudioSpectrogram',
    'TriangleFiltering',
    'DCT',
    'ClipCepstrum',
    'TrimmingAudio',
    'SamplesToFloat32',
    'AudioToMelSpectrogram',

    'Resize',
    'Resize3D',
    'AutoResize',
    'Flip',
    'PointAligner',
    'Tiling',
    'CropBraTS',
    'ImagePyramid',
    'FaceDetectionImagePyramid',
    'WarpAffine',
    'BgrToGray',
    'BgrToRgb',
    'RgbToGray',
    'RgbToBgr',
    'BGR2YUVConverter',
    'BGRToLAB',
    'RGBToLAB',
    'TfConvertImageDType',
    'SelectInputChannel',
    'CropOrPad',
    'CropWithPadSize',
    'Crop',
    'CornerCrop',
    'CandidateCrop',
    'CropRect',
    'Crop3D',
    'ObjectCropWithScale',

    'Normalize3d',
    'Normalize',
    'NormalizeBrats',

    'SwapModalitiesBrats',

    'DecodeByVocabulary',
    'PadWithEOS',

    'CenterNetAffineTransformation',

    'FreeFormMask',
    'RectMask',
    'CustomMask',
    'RGB2YCrCbConverter',
    'BGR2YCrCbConverter',
    'OneHotEncoding'
]
