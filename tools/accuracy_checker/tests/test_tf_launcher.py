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
pytest.importorskip('tensorflow')

import cv2
import numpy as np

from accuracy_checker.launcher.launcher import create_launcher
from accuracy_checker.config import ConfigError


def no_tf2():
    try:
        import tensorflow
        return tensorflow.__version__ < '2.0.0'
    except ImportError:
        return True

def get_tf_test_model(models_dir, config_update=None):
    config = {
        "framework": "tf",
        "model": str(models_dir / "samplenet.pb"),
        "adapter": 'classification',
        "device": "cpu"
    }
    if config_update:
        config.update(config_update)

    return create_launcher(config)


def get_tf2_test_model(models_dir, config_update=None):
    config = {
        "framework": "tf2",
        "saved_model_dir": str(models_dir / "samplenet_tf2"),
        "adapter": 'classification',
        "device": "cpu"
    }
    if config_update:
        config.update(config_update)

    return create_launcher(config)


class TestTFLauncher:
    def test_launcher_creates(self, models_dir):
        tf_test_model = get_tf_test_model(models_dir)
        assert tf_test_model.inputs['data'] == (1, 3, 32, 32)
        assert tf_test_model.output_blob == 'add_4'

    def test_infer(self, data_dir, models_dir):
        tf_test_model = get_tf_test_model(models_dir)
        _, _, h, w = tf_test_model.inputs['data']
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h))
        # Test model was converted from mxnet. Original layout was saved.
        input_blob = np.transpose([img_resized], (0, 3, 1, 2))
        res = tf_test_model.predict([{'data': input_blob.astype(np.float32)}], [{}])

        assert np.argmax(res[0][tf_test_model.output_blob]) == 7


@pytest.mark.skipif(no_tf2(), reason='tensorflow version lower than required')
class TestTF2Launcher:
    def test_launcher_creates(self, models_dir):
        tf_test_model = get_tf2_test_model(models_dir)
        assert tf_test_model.inputs['conv2d_9_input']['shape'] == (None, 32, 32, 3)
        assert tf_test_model.output_blob == 'dense_12'
        tf_test_model.release()

    def test_infer(self, data_dir, models_dir):
        tf_test_model = get_tf2_test_model(models_dir)
        _, h, w, _ = tf_test_model.inputs['conv2d_9_input']['shape']
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h)).astype(np.float32) / 255
        input_blob = tf_test_model.fit_to_input([img_resized], 'conv2d_9_input', [0, 1, 2, 3], np.float32)
        res = tf_test_model.predict([{'conv2d_9_input': input_blob}], [{}])

        assert np.argmax(res[0][tf_test_model.output_blob]) == 2


class TestTFLauncherConfig:
    def test_missed_model_in_tf_launcher_config_raises_config_error_exception(self):
        launcher = {'framework': 'tf', 'adapter': 'classification'}

        with pytest.raises(ConfigError):
            create_launcher(launcher)

    @pytest.mark.usefixtures('mock_path_exists')
    def test_unknown_device_in_tf_launcher_config_raises_config_error_exception(self):
        launcher = {'framework': 'tf', 'adapter': 'classification', 'model': 'custom', 'device': 'unknown'}

        with pytest.raises(ConfigError):
            create_launcher(launcher)

    def test_unknown_output_name_in_create_tf_launcher_raises_config_error_exception(self, models_dir):
        with pytest.raises(ConfigError):
            get_tf_test_model(models_dir, {'output_names': ['name']})


@pytest.mark.skipif(no_tf2(), reason='tensorflow version lower than required')
class TestTF2LauncherConfig:
    def test_missed_saved_model_dir_in_tf2_launcher_config_raises_config_error_exception(self):
        launcher = {'framework': 'tf2', 'adapter': 'classification'}

        with pytest.raises(ConfigError):
            create_launcher(launcher)

    @pytest.mark.usefixtures('mock_path_exists')
    def test_unknown_device_in_tf2_launcher_config_raises_config_error_exception(self):
        launcher = {'framework': 'tf2', 'adapter': 'classification', 'saved_model_dir': 'custom', 'device': 'unknown'}

        with pytest.raises(ConfigError):
            create_launcher(launcher)
