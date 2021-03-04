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

import pytest
pytest.importorskip('accuracy_checker.launcher.opencv_launcher')

import cv2
import numpy as np

from accuracy_checker.launcher.launcher import create_launcher
from accuracy_checker.config import ConfigError


def get_opencv_test_caffe_model(models_dir):
    config = {
        "framework": "opencv",
        "weights": str(models_dir / "SampLeNet.caffemodel"),
        "model": str(models_dir / "SampLeNet.prototxt"),
        "adapter": "classification",
        "device": "cpu",
        "backend": "ocv",
        "inputs": [{"name": "input", "type": "INPUT", "shape": "(3, 32, 32)"}]
    }
    return create_launcher(config)


def get_opencv_test_onnx_model(models_dir):
    config = {
        "framework": "opencv",
        "model": str(models_dir / "samplenet.onnx"),
        "adapter": "classification",
        "device": "cpu",
        "backend": "ocv",
        "inputs": [{"name": "input", "type": "INPUT", "shape": "(3, 32, 32)"}]
    }
    return create_launcher(config)


list_models = [get_opencv_test_caffe_model, get_opencv_test_onnx_model]


class TestOpenCVLauncher:
    def test_launcher_creates(self, models_dir):
        for model_element in list_models:
            assert model_element(models_dir).inputs['input'] == (1, 3, 32, 32)

    def test_infer_caffe_model(self, data_dir, models_dir):
        opencv_test_model = get_opencv_test_caffe_model(models_dir)
        _, _, h, w = opencv_test_model.inputs['input']
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_resized = cv2.resize(img_raw, (w, h))
        input_blob = np.transpose([img_resized], (0, 3, 1, 2))
        res = opencv_test_model.predict([{'input': input_blob.astype(np.float32)}], [{}])
        assert np.argmax(res[0]['fc3']) == 7

    def test_infer_onnx_model(self, data_dir, models_dir):
        opencv_test_model = get_opencv_test_onnx_model(models_dir)
        _, _, h, w = opencv_test_model.inputs['input']
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h))
        input_blob = np.transpose([img_resized], (0, 3, 1, 2))
        res = opencv_test_model.predict([{'input': input_blob.astype(np.float32)}], [{}])
        assert np.argmax(res[0]['fc3']) == 7


@pytest.mark.usefixtures('mock_path_exists')
class TestOpenCVLauncherConfig:
    def test_missed_framework_in_create_opencv_launcher_raises_config_error_exception(self):
        config = {
            # 'framework': 'opencv',
            'model': 'model.ocv',
            'weights': 'weights.ocv',
            'device': 'not_device',
            'backend': 'not_backend',
            'adapter': 'classification',
            'inputs': [{'name': 'data', 'type': 'INPUT'}]
        }
        with pytest.raises(KeyError):
            create_launcher(config)

    def test_missed_model_in_create_opencv_launcher_raises_config_error_exception(self):
        config = {
            'framework': 'opencv',
            # 'model': 'model.ocv',
            'weights': 'weights.ocv',
            'device': 'not_device',
            'backend': 'not_backend',
            'adapter': 'classification',
            'inputs': [{'name': 'data', 'type': 'INPUT'}]
        }
        with pytest.raises(ConfigError):
            create_launcher(config, 'model')

    def test_missed_device_in_create_opencv_launcher_raises_config_error_exception(self):
        config = {
            'framework': 'opencv',
            'model': 'model.ocv',
            'weights': 'weights.ocv',
            # 'device': 'not_device',
            'backend': 'not_backend',
            'adapter': 'classification',
            'inputs': [{'name': 'data', 'type': 'INPUT'}]
        }
        with pytest.raises(ConfigError):
            create_launcher(config)

    def test_missed_backend_in_create_opencv_launcher_raises_config_error_exception(self):
        config = {
            'framework': 'opencv',
            'model': 'model.ocv',
            'weights': 'weights.ocv',
            'device': 'not_device',
            # 'backend': 'not_backend',
            'adapter': 'classification',
            'inputs': [{'name': 'data', 'type': 'INPUT'}]
        }
        with pytest.raises(ConfigError):
            create_launcher(config)

    def test_missed_adapter_in_create_opencv_launcher_raises_config_error_exception(self):
        config = {
            'framework': 'opencv',
            'model': 'model.ocv',
            'weights': 'weights.ocv',
            'device': 'not_device',
            'backend': 'not_backend',
            # 'adapter': 'classification',
            'inputs': [{'name': 'data', 'type': 'INPUT'}]
        }
        with pytest.raises(ConfigError):
            create_launcher(config)

    def test_missed_inputs_in_create_opencv_launcher_raises_config_error_exception(self):
        config = {
            'framework': 'opencv',
            'model': 'model.ocv',
            'weights': 'weights.ocv',
            'device': 'not_device',
            'backend': 'not_backend',
            'adapter': 'classification',
            # 'inputs': [{'name': 'data', 'type': 'INPUT'}]
        }
        with pytest.raises(ConfigError):
            create_launcher(config)
