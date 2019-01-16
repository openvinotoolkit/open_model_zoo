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

from .adapter import Adapter, AdapterField

from .common_adapters import (TinyYOLOv1Adapter, ClassificationAdapter, ReidAdapter,
                              LPRAdapter, VehicleAttributesRecognitionAdapter, HeadPoseEstimatorAdapter,
                              FacePersonAdapter, SSDAdapter, PersonAttributesAdapter, ActionDetection,
                              SuperResolutionAdapter)

from .dummy_adapters import XML2DetectionAdapter

__all__ = [
    'Adapter',
    'AdapterField',
    'ReidAdapter',
    'ClassificationAdapter',
    'TinyYOLOv1Adapter',
    'XML2DetectionAdapter',
    'LPRAdapter',
    'FacePersonAdapter',
    'VehicleAttributesRecognitionAdapter',
    'SSDAdapter',
    'XML2DetectionAdapter',
    'LPRAdapter',
    'HeadPoseEstimatorAdapter',
    'PersonAttributesAdapter',
    'ActionDetection',
    'SuperResolutionAdapter'
]
