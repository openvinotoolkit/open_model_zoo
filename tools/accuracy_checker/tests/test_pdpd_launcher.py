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


def get_pdpd_test_model(models_dir):
    config = {
        "framework": "paddle_paddle",
        "model": str(models_dir / "samplenet.pdmodel"),
        'params': str(models_dir / 'samplenet.pdiparams'),
        "adapter": "classification",
    }
    return create_launcher(config)


class TestPaddlePaddleLauncher:
    def test_launcher_creates(self, models_dir):
        launcher = get_pdpd_test_model(models_dir)
        assert launcher.inputs['x'] == [-1, 3, 32, 32]
        assert launcher.output_blob == 'save_infer_model/scale_0.tmp_0'

    def test_infer(self, data_dir, models_dir):
        pdpd_test_model = get_pdpd_test_model(models_dir)
        _, _, h, w = pdpd_test_model.inputs['x']
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h))
        input_dict = {
            'x': pdpd_test_model.fit_to_input([img_resized.astype(np.float32) / 255], 'x', [0, 3, 1, 2], np.float32)
        }
        res = pdpd_test_model.predict([input_dict], [{}])

        assert np.argmax(res[0]['save_infer_model/scale_0.tmp_0']) == 7

@pytest.mark.usefixtures('mock_path_exists')
class TestPaddlePaddleLauncherConfig:
    def test_missed_model_in_create_pdpd_launcher_raises_config_error_exception(self):
        config = {'framework': 'paddle_paddle'}

        with pytest.raises(ConfigError):
            create_launcher(config)

    def test_unsupported_device_in_create_onnx_launcher_raises_config_error_exception(self):
        config = {
            'framework': 'paddle_paddle', 'model': 'model.pdmodel', 'params': 'params.pdiparams',
            'device': 'UNSUPPORTED'}

        with pytest.raises(ConfigError):
            create_launcher(config)
