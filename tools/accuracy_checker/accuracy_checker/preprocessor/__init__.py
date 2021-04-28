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
from .audio_preprocessing_ext import (
    SpliceFrame,
    DitherFrame,
    DitherSpectrum,
    PreemphFrame,
    SignalPatching,
    ContextWindow
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
from .raw_image_preprocessing import PackBayerImage
from .trimap import TrimapPreprocessor, AlphaChannel

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
    'SpliceFrame',
    'DitherFrame',
    'DitherSpectrum',
    'PreemphFrame',
    'SignalPatching',
    'ContextWindow',

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
    'RGB2YUVConverter',
    'BGRtoNV12Converter',
    'RGBtoNV12Converter',
    'NV12toBGRConverter',
    'NV12toRGBConverter',
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
    'ExtendAroundRect',
    'Crop3D',
    'TransformedCropWithAutoScale',
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
    'OneHotEncoding',

    'PackBayerImage',

    'TrimapPreprocessor',
    'AlphaChannel'
]
