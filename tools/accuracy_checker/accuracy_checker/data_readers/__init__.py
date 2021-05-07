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

from .data_reader import (
    BaseReader,
    DataReaderField,
    ReaderCombiner,
    OpenCVFrameReader,
    OpenCVImageReader,
    PillowImageReader,
    ScipyImageReader,
    NiftiImageReader,
    NumPyReader,
    NumpyTXTReader,
    NumpyDictReader,
    NumpyBinReader,
    TensorflowImageReader,
    AnnotationFeaturesReader,
    WavReader,
    DicomReader,
    PickleReader,
    SkimageReader,
    RawpyReader,
    ByteFileReader,
    LMDBReader,
    KaldiARKReader,

    DataRepresentation,
    ClipIdentifier,
    MultiFramesInputIdentifier,
    ImagePairIdentifier,
    ListIdentifier,
    MultiInstanceIdentifier,
    KaldiFrameIdentifier,
    KaldiMatrixIdentifier,
    ParametricImageIdentifier,

    serialize_identifier,
    deserialize_identifier,
    create_identifier_key,

    create_reader,
    REQUIRES_ANNOTATIONS
)

__all__ = [
    'BaseReader',
    'DataReaderField',
    'DataRepresentation',
    'ReaderCombiner',
    'OpenCVFrameReader',
    'OpenCVImageReader',
    'PillowImageReader',
    'ScipyImageReader',
    'NiftiImageReader',
    'TensorflowImageReader',
    'AnnotationFeaturesReader',
    'WavReader',
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

    'DataRepresentation',
    'ClipIdentifier',
    'MultiFramesInputIdentifier',
    'ImagePairIdentifier',
    'ListIdentifier',
    'MultiInstanceIdentifier',
    'KaldiMatrixIdentifier',
    'KaldiFrameIdentifier',
    'ParametricImageIdentifier',

    'create_reader',
    'REQUIRES_ANNOTATIONS',

    'serialize_identifier',
    'deserialize_identifier',
    'create_identifier_key'
]
