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
pytest.importorskip('accuracy_checker.launcher.caffe_launcher')

import cv2
import numpy as np

from accuracy_checker.launcher.launcher import create_launcher
from accuracy_checker.config import ConfigError
from accuracy_checker.data_readers import DataRepresentation


def get_caffe_test_model(models_dir):
    config = {
        "framework": "caffe",
        "weights": str(models_dir / "SampLeNet.caffemodel"),
        "model": str(models_dir / "SampLeNet.prototxt"),
        "adapter": 'classification',
        "device": "cpu"
    }

    return create_launcher(config)


class TestCaffeLauncher:
    def test_launcher_creates(self, models_dir):
        assert get_caffe_test_model(models_dir).inputs['data'] == (1, 3, 32, 32)

    def test_infer(self, data_dir, models_dir):
        caffe_test_model = get_caffe_test_model(models_dir)
        _, _, h, w = caffe_test_model.inputs['data']
        img_raw = cv2.imread(str(data_dir / '1.jpg'))
        img_resized = cv2.resize(img_raw, (w, h))
        input_blob = np.transpose([img_resized], (0, 3, 1, 2))
        res = caffe_test_model.predict([{'data': input_blob.astype(np.float32)}], [{}])

        assert np.argmax(res[0]['fc3']) == 7

    def test_caffe_launcher_provide_input_shape_to_adapter(self, mocker, models_dir):
        mocker.patch('caffe.Net.forward', return_value={'fc3': 0})
        launcher = get_caffe_test_model(models_dir)
        zeros = DataRepresentation(np.zeros((1, 3, 32, 32)))
        launcher.predict([{'data': zeros.data}], [zeros.metadata])
        assert zeros.metadata['input_shape'] == {'data': (1, 3, 32, 32)}

    def test_caffe_launcher_model_search(self, models_dir):
        config = {
            "framework": "caffe",
            "weights": models_dir,
            "model": models_dir,
            "adapter": 'classification',
            "device": "cpu"
        }
        caffe_model = create_launcher(config, 'SampleNet')
        assert caffe_model.model == models_dir / 'SampLeNet.prototxt'
        assert caffe_model.weights == models_dir / 'SampLeNet.caffemodel'


def test_missed_model_in_create_caffe_launcher_raises_config_error_exception():
    launcher = {'framework': 'caffe', 'weights': 'custom', 'adapter': 'classification'}

    with pytest.raises(ConfigError):
        create_launcher(launcher)


def test_missed_weights_in_create_caffe_launcher_raises_config_error_exception():
    launcher = {'framework': 'caffe', 'model': 'custom', 'adapter': 'ssd'}

    with pytest.raises(ConfigError):
        create_launcher(launcher)
