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

import numpy as np
import pytest

pytest.importorskip('accuracy_checker.launcher.openvino_launcher')

from accuracy_checker.launcher.openvino_launcher import OpenVINOLauncher


def test_npu_initializes_unbounded_dynamic_shape_before_compilation(mocker):
    launcher = OpenVINOLauncher.__new__(OpenVINOLauncher)
    launcher._device = 'NPU'
    launcher._partial_shapes = {'pixel_values': '[-1,3,-1,-1]'}
    launcher.dynamic_shapes_policy = 'dynamic'
    launcher.is_dynamic = True
    launcher._reshape_input = mocker.Mock()

    launcher.initialize_undefined_shapes([{'pixel_values': np.zeros((1, 3, 224, 224), dtype=np.float32)}])

    launcher._reshape_input.assert_called_once_with({'pixel_values': (1, 3, 224, 224)})
    assert not launcher.is_dynamic


def test_npu_keeps_bounded_dynamic_shape_initialization(mocker):
    launcher = OpenVINOLauncher.__new__(OpenVINOLauncher)
    launcher._device = 'NPU'
    launcher._partial_shapes = {'pixel_values': '[1..4,3,224..512,224..512]'}
    launcher.dynamic_shapes_policy = 'dynamic'
    launcher.is_dynamic = True
    launcher.load_network = mocker.Mock()
    launcher.exec_network = mocker.Mock()
    launcher.input_to_tensor_name = {'pixel_values': 'pixel_values'}

    launcher.initialize_undefined_shapes([{'pixel_values': np.zeros((1, 3, 224, 224), dtype=np.float32)}])

    launcher.load_network.assert_not_called()
    launcher.exec_network.infer_new_request.assert_called_once()