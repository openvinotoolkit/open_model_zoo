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
pytest.importorskip('accuracy_checker.launcher.onnx_launcher')

import cv2
import numpy as np

from accuracy_checker.launcher.launcher import create_launcher
from accuracy_checker.config import ConfigError


def old_onnxrunitme(models_dir):
    import onnxruntime as rt
    sess = rt.InferenceSession(str(models_dir / "samplenet.onnx"))
    try:
        sess.get_providers()
        return False
    except AttributeError:
        return True


def get_onnx_test_model(models_dir, device=None, ep=None):
    config = {
        "framework": "onnx_runtime",
        "model": str(models_dir / "samplenet.onnx"),
        "adapter": "classification",
    }
    if device is not None:
        config['device'] = device
    if ep is not None:
        config['execution_providers'] = ep
    return create_launcher(config)


class TestONNXRuntimeLauncher:
    def test_launcher_creates(self, models_dir):
        launcher = get_onnx_test_model(models_dir)
        assert launcher.inputs['data'] == [1, 3, 32, 32]
        assert launcher.output_blob == 'fc3'

    def test_infer(self, data_dir, models_dir):
        onnx_test_model = get_onnx_test_model(models_dir)
        _, _, h, w = onnx_test_model.inputs['data']
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h))
        input_blob = np.transpose([img_resized], (0, 3, 1, 2))
        res = onnx_test_model.predict([{'data': input_blob.astype(np.float32)}], [{}])

        assert np.argmax(res[0]['fc3']) == 7

    def test_infer_with_execution_provider(self, data_dir, models_dir):
        if old_onnxrunitme(models_dir):
            pytest.skip(reason="onnxruntime does not support EP")
        onnx_test_model = get_onnx_test_model(models_dir, ep=['CPUExecutionProvider'])
        _, _, h, w = onnx_test_model.inputs['data']
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h))
        input_blob = np.transpose([img_resized], (0, 3, 1, 2))
        res = onnx_test_model.predict([{'data': input_blob.astype(np.float32)}], [{}])

        assert np.argmax(res[0]['fc3']) == 7

    def test_auto_model_search(self, models_dir):
        config = {
            "framework": "onnx_runtime",
            "model": models_dir,
        }
        launcher = create_launcher(config, 'samplenet')
        assert launcher.model == models_dir / "samplenet.onnx"


@pytest.mark.usefixtures('mock_path_exists')
class TestONNXRuntimeLauncherConfig:
    def test_missed_model_in_create_onnx_launcher_raises_config_error_exception(self):
        config = {'framework': 'onnx_runtime'}

        with pytest.raises(ConfigError):
            create_launcher(config)

    def test_unsupported_device_in_create_onnx_launcher_raises_config_error_exception(self):
        config = {'framework': 'onnx_runtime', 'model': 'model.onnx', 'device': 'UNSUPPORTED'}

        with pytest.raises(ConfigError):
            create_launcher(config)
