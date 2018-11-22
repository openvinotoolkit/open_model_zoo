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
from .pascal_voc import PascalVOCDetectionConverter
from .sample_converter import SampleConverter
from .wider import WiderFormatConverter
from .detection_opencv_storage import DetectionOpenCVStorageFormatConverter
from .lfw import FaceReidPairwiseConverter
from .vgg_face_regression import LandmarksRegression

__all__ = [
    'SampleConverter',
    'PascalVOCDetectionConverter',
    'WiderFormatConverter',
    'DetectionOpenCVStorageFormatConverter',
    'FaceReidPairwiseConverter',
    'LandmarksRegression',
]
