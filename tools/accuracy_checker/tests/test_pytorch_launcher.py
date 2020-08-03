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

import pytest
pytest.importorskip('torch')
import cv2
import numpy as np

from accuracy_checker.launcher.launcher import create_launcher
from accuracy_checker.config import ConfigError

def get_pth_test_model(models_dir):
    config = {
        "framework": 'pytorch',
        "module": 'samplenet.SampLeNet',
        "checkpoint": models_dir/'pytorch_model'/'samplenet.pth',
        'python_path': models_dir/'pytorch_model',
        "adapter": 'classification',
        "device": 'cpu',
    }

    return create_launcher(config)


class TestPytorchLauncher:
    def test_launcher_creates(self, models_dir):
        launcher = get_pth_test_model(models_dir)
        assert launcher.inputs['input'] == (1, -1, -1, -1)
        assert launcher.output_blob == 'output'

    def test_infer(self, data_dir, models_dir):
        pytorch_test_model = get_pth_test_model(models_dir)
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_resized = cv2.resize(img_raw, (32, 32))
        rgb_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_blob = pytorch_test_model.fit_to_input([rgb_image], 'input', (0, 3, 1, 2))

        res = pytorch_test_model.predict([{'input': input_blob}], [{}])

        assert np.argmax(res[0]['output']) == 5


@pytest.mark.usefixtures('mock_path_exists')
class TestMxNetLauncherConfig:
    def test_missed_model_in_create_pytoch_launcher_raises_config_error_exception(self):
        config = {'framework': 'pytorch'}

        with pytest.raises(ConfigError):
            create_launcher(config)
