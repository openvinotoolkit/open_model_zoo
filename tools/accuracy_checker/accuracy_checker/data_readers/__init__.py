"""
Copyright (c) 2018-2024 Intel Corporation

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

from .data_reader import (
    BaseReader,
    DataReaderField,
    ReaderCombiner,
    DataRepresentation,
    AnnotationDataIdentifier,
    ClipIdentifier,
    MultiFramesInputIdentifier,
    ImagePairIdentifier,
    ListIdentifier,
    MultiInstanceIdentifier,
    KaldiFrameIdentifier,
    KaldiMatrixIdentifier,
    ParametricImageIdentifier,
    VideoFrameIdentifier,

    serialize_identifier,
    deserialize_identifier,
    create_identifier_key,
    create_ann_identifier_key,

    create_reader,
    REQUIRES_ANNOTATIONS
)
from .annotation_readers import AnnotationFeaturesReader
from .binary_data_readers import PickleReader, ByteFileReader, LMDBReader
from .medical_imaging_readers import NiftiImageReader, DicomReader
from .audio_readers import WavReader, KaldiARKReader, FlacReader
from .numpy_readers import NumPyReader, NumpyTXTReader, NumpyDictReader, NumpyBinReader
from .image_readers import (
    OpenCVImageReader,
    PillowImageReader,
    ScipyImageReader,
    OpenCVFrameReader,
    TensorflowImageReader,
    SkimageReader,
    RawpyReader
)
from .text_readers import JSONReader

__all__ = [
    'BaseReader',
    'DataReaderField',
    'DataRepresentation',
    'ReaderCombiner',
    'DataRepresentation',
    'ClipIdentifier',
    'MultiFramesInputIdentifier',
    'ImagePairIdentifier',
    'ListIdentifier',
    'MultiInstanceIdentifier',
    'KaldiMatrixIdentifier',
    'KaldiFrameIdentifier',
    'ParametricImageIdentifier',
    'VideoFrameIdentifier',
    'AnnotationDataIdentifier',

    'OpenCVFrameReader',
    'OpenCVImageReader',
    'PillowImageReader',
    'ScipyImageReader',
    'NiftiImageReader',
    'TensorflowImageReader',
    'AnnotationFeaturesReader',
    'WavReader',
    'FlacReader',
    'DicomReader',
    'PickleReader',
    'NumPyReader',
    'NumpyTXTReader',
    'NumpyDictReader',
    'NumpyBinReader',
    'SkimageReader',
    'RawpyReader',
    'ByteFileReader',
    'LMDBReader',
    'KaldiARKReader',
    'JSONReader',

    'create_reader',
    'REQUIRES_ANNOTATIONS',

    'serialize_identifier',
    'deserialize_identifier',
    'create_identifier_key',
    'create_ann_identifier_key'
]
