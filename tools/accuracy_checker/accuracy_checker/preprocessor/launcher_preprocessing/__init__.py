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

from .ie_preprocessor import IEPreprocessor, ie_preprocess_available
from .ov_preprocessor import OVPreprocessor, ov_preprocess_available
from .gapi_preprocessor import GAPIPreprocessor, gapi_preprocess_available

__all__ = [
    'preprocessing_available',
    'get_preprocessor'
]

FRAMEWORK_PREPROCESSINGS = {
    'gapi': GAPIPreprocessor,
    'openvino': OVPreprocessor,
    'dlsdk': IEPreprocessor
}

AVAILABILITY_CHECK = {
    'gapi': gapi_preprocess_available,
    'openvino': ov_preprocess_available,
    'dlsdk': ie_preprocess_available
}


def preprocessing_available(framework):
    if framework not in AVAILABILITY_CHECK:
        return False
    return AVAILABILITY_CHECK[framework]()


def get_preprocessor(framework):
    return FRAMEWORK_PREPROCESSINGS[framework]
