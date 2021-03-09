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
pytest.importorskip('mxnet')
import cv2
import numpy as np

from accuracy_checker.launcher.launcher import create_launcher
from accuracy_checker.config import ConfigError
from accuracy_checker.data_readers import DataRepresentation


def get_mx_test_model(models_dir, config_override=None):
    config = {
        "framework": 'mxnet',
        "model": models_dir / 'samplenet-0000.params',
        "adapter": 'classification',
        "device": 'cpu',
        'inputs': [{'name': 'data', 'type': 'INPUT', 'shape': '3,32,32'}]
    }
    if config_override:
        config.update(config_override)

    return create_launcher(config)


class TestMxNetLauncher:
    def test_launcher_creates(self, models_dir):
        launcher = get_mx_test_model(models_dir)
        assert launcher.inputs['data'] == (1, 3, 32, 32)
        assert launcher.output_blob == 'fc3'

    def test_infer(self, data_dir, models_dir):
        mx_test_model = get_mx_test_model(models_dir)
        _, _, h, w = mx_test_model.inputs['data']
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h))
        input_blob = np.transpose([img_resized], (0, 3, 1, 2))
        res = mx_test_model.predict([{'data': input_blob.astype(np.float32)}], [{}])

        assert np.argmax(res[0]['fc3']) == 7

    def test_mxnet_launcher_provide_input_shape_to_adapter(self, mocker, models_dir):
        mocker.patch('mxnet.mod.Module.forward', return_value={'fc3': 0})
        launcher = get_mx_test_model(models_dir)
        zeros = DataRepresentation(np.zeros((1, 3, 32, 32)))
        launcher.predict([{'data': zeros.data}], [zeros.metadata])
        assert zeros.metadata['input_shape'] == {'data': (1, 3, 32, 32)}

    def test_mxnet_launcher_auto_model_search(self, models_dir):
        launcher = get_mx_test_model(models_dir, {'model': models_dir})
        assert launcher.model == models_dir / 'samplenet-0000.params'


@pytest.mark.usefixtures('mock_path_exists')
class TestMxNetLauncherConfig:
    def test_missed_model_in_create_mxnet_launcher_raises_config_error_exception(self):
        config = {'framework': 'mxnet'}

        with pytest.raises(ConfigError):
            create_launcher(config)

    def test_missed_inputs_in_create_mxnet_launcher_raises_config_error_exception(self):
        config = {'framework': 'mxnet', 'model': 'model-0000.params'}

        with pytest.raises(ConfigError):
            create_launcher(config)

    def test_missed_shape_in_inputs_in_create_mxnet_launcher_raises_config_error_exception(self):
        config = {'framework': 'mxnet', 'model': 'model-0000.params', 'inputs': [{'name': 'data', 'type': 'INPUT'}]}

        with pytest.raises(ConfigError):
            create_launcher(config)
