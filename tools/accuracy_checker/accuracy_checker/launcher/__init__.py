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

from .dummy_launcher import DummyLauncher
from .launcher import Launcher, create_launcher, unsupported_launcher
from .input_feeder import InputFeeder

try:
    from .caffe_launcher import CaffeLauncher
except ImportError as import_error:
    CaffeLauncher = unsupported_launcher(
        'caffe', "Caffe isn't installed. Please, install it before using. \n{}".format(import_error.msg)
    )

try:
    from .mxnet_launcher import MxNetLauncher
except ImportError as import_error:
    MxNetLauncher = unsupported_launcher(
        'mxnet', "MXNet isn't installed. Please, install it before using.\n{}".format(import_error.msg)
    )

try:
    from .dlsdk_launcher import DLSDKLauncher
except ImportError as import_error:
    DLSDKLauncher = unsupported_launcher(
        'dlsdk', "IE Python isn't installed. Please, install it before using. \n{}".format(import_error.msg)
    )

try:
    from .tf_launcher import TFLauncher
except ImportError as import_error:
    TFLauncher = unsupported_launcher(
        'tf', "TensorFlow isn't installed. Please, install it before using. \n{}".format(import_error.msg)
    )

try:
    from .tf_lite_launcher import TFLiteLauncher
except ImportError as import_error:
    TFLiteLauncher = unsupported_launcher(
        'tf_lite', "TensorFlow isn't installed. Please, install it before using. \n{}".format(import_error.msg)
    )

from .opencv_launcher import OpenCVLauncher

try:
    from .onnx_launcher import ONNXLauncher
except ImportError as import_error:
    ONNXLauncher = unsupported_launcher(
        'onnx_runtime', "ONNX Runtime isn't installed. Please, install it before using. \n{}".format(import_error.msg)
    )

from .pytorch_launcher import PyTorchLauncher

__all__ = [
    'create_launcher',
    'Launcher',
    'CaffeLauncher',
    'MxNetLauncher',
    'TFLauncher',
    'TFLiteLauncher',
    'DLSDKLauncher',
    'OpenCVLauncher',
    'ONNXLauncher',
    'PyTorchLauncher',
    'DummyLauncher',
    'InputFeeder'
]
