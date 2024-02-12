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

import pytest
pytest.importorskip('openvino.inference_engine')
pytest.importorskip('cv2.gapi.ie.params')
pytest.importorskip('accuracy_checker.launcher.gapi_launcher')

import cv2
import numpy as np

from accuracy_checker.launcher.launcher import create_launcher
from accuracy_checker.config import ConfigError


def get_gapi_test_model(models_dir):
    config = {
        "framework": "g-api",
        #"weights": str(models_dir / "SampLeNet.bin"),
        "model": models_dir,
        "adapter": "classification",
        "device": "cpu",
        "inputs": [{"name": "data", "type": "INPUT", "shape": "(3, 32, 32)"}],
        'outputs': ['fc3']
    }
    return create_launcher(config)


class TestGAPILauncher:
    def test_launcher_creates(self, models_dir):
        assert get_gapi_test_model(models_dir).inputs['data'] == (1, 3, 32, 32)

    def test_infer_model(self, data_dir, models_dir):
        test_model = get_gapi_test_model(models_dir)
        _, _, h, w = test_model.inputs['data']
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_resized = cv2.resize(img_raw, (w, h))
        res = test_model.predict([{'data': img_resized}], [{}])
        assert np.argmax(res[0]['fc3']) == 7


@pytest.mark.usefixtures('mock_path_exists')
class TestOpenCVLauncherConfig:
    def test_missed_framework_in_create_gapi_launcher_raises_config_error_exception(self):
        config = {
            # 'framework': 'g-api',
            'model': 'model.xml',
            'weights': 'weights.bin',
            'device': 'CPU',
            'adapter': 'classification',
            'inputs': [{'name': 'data', 'type': 'INPUT'}],
            'outputs': ['out']
        }
        with pytest.raises(KeyError):
            create_launcher(config)

    def test_missed_model_in_create_gapi_launcher_raises_config_error_exception(self):
        config = {
            'framework': 'g-api',
            # 'model': 'model.ocv',
            'weights': 'weights.bin',
            'device': 'CPU',
            'adapter': 'classification',
            'inputs': [{'name': 'data', 'type': 'INPUT'}],
            'outputs': ['out']
        }
        with pytest.raises(ConfigError):
            create_launcher(config, 'model')

    def test_missed_device_in_create_gapi_launcher_raises_config_error_exception(self):
        config = {
            'framework': 'g-api',
            'model': 'model.xml',
            'weights': 'weights.bin',
            # 'device': 'not_device',
            'adapter': 'classification',
            'inputs': [{'name': 'data', 'type': 'INPUT'}],
            'outputs': ['out']
        }
        with pytest.raises(ConfigError):
            create_launcher(config)


    def test_missed_inputs_in_create_gapi_launcher_raises_config_error_exception(self):
        config = {
            'framework': 'g-api',
            'model': 'model.xml',
            'weights': 'weights.bin',
            'device': 'CPU',
            'backend': 'not_backend',
            'adapter': 'classification',
            # 'inputs': [{'name': 'data', 'type': 'INPUT'}]
            'outputs': ['out']
        }
        with pytest.raises(ConfigError):
            create_launcher(config)

    def test_missed_outputs_in_create_gapi_launcher_raises_config_error_exception(self):
        config = {
            'framework': 'g-api',
            'model': 'model.xml',
            'weights': 'weights.bin',
            'device': 'CPU',
            'backend': 'not_backend',
            'adapter': 'classification',
            'inputs': [{'name': 'data', 'type': 'INPUT'}]
            #'outputs': ['out']
        }
        with pytest.raises(ConfigError):
            create_launcher(config)
