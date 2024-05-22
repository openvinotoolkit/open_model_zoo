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

pytest.importorskip('accuracy_checker.launcher.dlsdk_launcher')
import cv2
import numpy as np

from pathlib import Path
from unittest.mock import PropertyMock
from accuracy_checker.config import ConfigError
from accuracy_checker.launcher import DLSDKLauncher
from accuracy_checker.launcher.dlsdk_launcher_config import DLSDKLauncherConfigValidator
from accuracy_checker.launcher.launcher import create_launcher
from tests.common import update_dict
from accuracy_checker.data_readers import DataRepresentation
from accuracy_checker.utils import contains_all
try:
    import ngraph as ng
except ImportError:
    ng = None


def no_available_myriad():
    return True


def no_available_gpu():
    from openvino import Core
    return 'GPU' not in Core().available_devices


def has_layers():
    try:
        from openvino.inference_engine import IENetwork
        return hasattr(IENetwork, 'layers')
    except Exception:
        return False


@pytest.fixture()
def mock_inference_engine(mocker):
    mocker.patch('openvino.inference_engine.IECore')
    mocker.patch('openvino.inference_engine.IENetwork')


@pytest.fixture()
def mock_inputs(mocker):
    mocker.patch(
        'accuracy_checker.launcher.input_feeder.InputFeeder._parse_inputs_config',
        return_value=({}, ['data'], None)
    )


@pytest.fixture()
def mock_affinity_map_exists(mocker):
    mocker.patch('pathlib.Path.exists', return_value=True)
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('pathlib.Path.is_file', return_value=True)

    def side_effect(filename):
        if filename == './affinity_map.yml':
            return True
        else:
            Path.exists(filename)
    mocker.side_effect = side_effect


def get_dlsdk_test_model(models_dir, config_update=None):
    config = {
        'framework': 'dlsdk',
        'weights': str(models_dir / 'SampLeNet.bin'),
        'model': str(models_dir / 'SampLeNet.xml'),
        'device': 'CPU',
        'adapter': 'classification',
    }
    if config_update:
        config.update(config_update)

    return create_launcher(config, model_name='SampLeNet')


def get_dlsdk_test_blob(models_dir, config_update=None):
    config = {
        'framework': 'dlsdk',
        'model': str(models_dir / 'SampLeNet.blob'),
        'device': 'MYRIAD',
        'adapter': 'classification',
    }
    if config_update:
        config.update(config_update)

    return create_launcher(config)


def get_onnx_test_model(model_dir, config_update=None):
    config = {
        'framework': 'dlsdk',
        'model': str(model_dir / 'samplenet.onnx'),
        'device': 'CPU',
        'adapter': 'classification'
    }
    if config_update:
        config.update(config_update)

    return create_launcher(config)


def get_image(image_path, input_shape):
    _, _, h, w = input_shape
    img_raw = cv2.imread(str(image_path))

    return DataRepresentation(cv2.resize(img_raw, (w, h)))


class TestDLSDKLauncherInfer:
    def test_infer(self, data_dir, models_dir):
        dlsdk_test_model = get_dlsdk_test_model(models_dir)
        image = get_image(data_dir / '1.jpg', dlsdk_test_model.inputs['data'].shape)
        input_blob = np.transpose([image.data], (0, 3, 1, 2))
        result = dlsdk_test_model.predict([{'data': input_blob.astype(np.float32)}], [image.metadata])
        assert dlsdk_test_model.output_blob == 'fc3'

        assert np.argmax(result[0][dlsdk_test_model.output_blob]) == 7
        assert image.metadata['input_shape'] == {'data': [1, 3, 32, 32]}

    def test_launcher_creates(self, models_dir):
        assert get_dlsdk_test_model(models_dir).inputs['data'].shape == [1, 3, 32, 32]

    def test_infer_with_additional_outputs(self, models_dir):
        dlsdk_test_model = get_dlsdk_test_model(models_dir, {'outputs': ['fc1', 'fc2']})
        outputs = list(dlsdk_test_model.network.outputs.keys())

        assert contains_all(outputs, ['fc1', 'fc2', 'fc3'])
        assert dlsdk_test_model.output_blob == 'fc3'

    def test_dlsdk_launcher_set_batch_size(self, models_dir):
        dlsdk_test_model = get_dlsdk_test_model(models_dir, {'batch': 2})
        assert dlsdk_test_model.batch == 2

    @pytest.mark.skipif(no_available_myriad(), reason='no myriad device in the system')
    def test_dlsdk_launcher_import_network(self, data_dir, models_dir):
        dlsdk_test_model = get_dlsdk_test_blob(models_dir)
        image = get_image(data_dir / '1.jpg', dlsdk_test_model.inputs['data'].shape)
        input_blob = np.transpose([image.data], (0, 3, 1, 2))
        result = dlsdk_test_model.predict([{'data': input_blob.astype(np.float32)}], [image.metadata])
        assert dlsdk_test_model.output_blob == 'fc3'

        assert np.argmax(result[0][dlsdk_test_model.output_blob]) == 7
        assert image.metadata['input_shape'] == {'data': [1, 3, 32, 32]}

    def test_dlsdk_launcher_model_search(self, models_dir):
        config_update = {
            'model': str(models_dir),
            'weights': str(models_dir)
        }
        dlsdk_test_model = get_dlsdk_test_model(models_dir, config_update)
        assert dlsdk_test_model._model == models_dir / 'SampLeNet.xml'
        assert dlsdk_test_model._weights == models_dir / 'SampLeNet.bin'

    def test_dlsdk_onnx_import(self, data_dir, models_dir):
        dlsdk_test_model = get_onnx_test_model(models_dir)
        image = get_image(data_dir / '1.jpg', dlsdk_test_model.inputs['data'].shape)
        input_blob = np.transpose([image.data], (0, 3, 1, 2))
        dlsdk_test_model.predict([{'data': input_blob.astype(np.float32)}], [image.metadata])
        assert dlsdk_test_model.output_blob == 'fc3'


@pytest.mark.skipif(ng is None and not has_layers(), reason='no functionality to set affinity')
class TestDLSDKLauncherAffinity:
    @pytest.mark.skipif(no_available_gpu(), reason='no GPU')
    @pytest.mark.usefixtures('mock_affinity_map_exists')
    def test_dlsdk_launcher_valid_affinity_map(self, mocker, models_dir):
        affinity_map = {'conv1': 'GPU'}
        if not has_layers():
            affinity_map.update({
                'conv1/Dims294/copy_const': 'GPU'
            })

        mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.read_yaml', return_value=affinity_map
        )

        dlsdk_test_model = get_dlsdk_test_model(models_dir, {
            'device': 'HETERO:CPU,GPU', 'affinity_map': './affinity_map.yml'
        })
        if has_layers():
            layers = dlsdk_test_model.network.layers
            for key, value in affinity_map.items():
                assert layers[key].affinity == value
        else:
            ng_function = ng.function_from_cnn(dlsdk_test_model.network)
            for node in ng_function.get_ordered_ops():
                if node.get_friendly_name() != 'conv1':
                    continue
                assert node.get_friendly_name() in affinity_map
                assert node.get_rt_info()['affinity'] == affinity_map[node.get_friendly_name()]

    @pytest.mark.usefixtures('mock_file_exists')
    def test_dlsdk_launcher_affinity_map_invalid_device(self, mocker, models_dir):
        affinity_map = {'conv1': 'GPU'}

        mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.read_yaml', return_value=affinity_map
        )

        with pytest.raises(ConfigError):
            get_dlsdk_test_model(models_dir, {'device': 'HETERO:CPU,CPU', 'affinity_map': './affinity_map.yml'})

    @pytest.mark.usefixtures('mock_file_exists')
    def test_dlsdk_launcher_affinity_map_invalid_layer(self, mocker, models_dir):
        affinity_map = {'none-existing-layer': 'CPU'}

        mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.read_yaml', return_value=affinity_map
        )

        with pytest.raises(ConfigError):
            get_dlsdk_test_model(models_dir, {'device': 'HETERO:CPU,CPU', 'affinity_map': './affinity_map.yml'})


@pytest.mark.usefixtures('mock_path_exists', 'mock_inference_engine', 'mock_inputs')
class TestDLSDKLauncher:
    def test_dlsdk_launcher_device_config_config_not_dict_like(self, models_dir):
        device_config = 'ENFORCE_BF16'

        with pytest.raises(ConfigError):
            get_dlsdk_test_model(models_dir, {'device_config': device_config})

    def test_dlsdk_launcher_device_config_device_unknown(self, models_dir):
        device_config = {'device': {'ENFORCE_BF16': 'NO'}}

        with pytest.warns(Warning):
            get_dlsdk_test_model(models_dir, {'device_config': device_config})

    def test_dlsdk_launcher_device_config_one_option_for_device_is_not_dict(self, models_dir):
        device_config = {'CPU': {'ENFORCE_BF16': 'NO'}, 'GPU': 'ENFORCE_BF16'}

        with pytest.warns(Warning):
            get_dlsdk_test_model(models_dir, {'device_config': device_config})

    def test_dlsdk_launcher_device_config_one_option_is_not_binding_to_device(self, models_dir):
        device_config = {'CPU': {'ENFORCE_BF16': 'NO'}, 'ENFORCE_BF16': 'NO'}

        with pytest.warns(Warning):
            get_dlsdk_test_model(models_dir, {'device_config': device_config})


@pytest.mark.usefixtures('mock_path_exists', 'mock_inputs', 'mock_inference_engine')
class TestDLSDKLauncherConfig:
    def setup(self):
        self.launcher = {
            'model': 'foo.xml',
            'weights': 'foo.bin',
            'device': 'CPU',
            'framework': 'dlsdk',
            'adapter': 'classification',
        }
        self.config = DLSDKLauncherConfigValidator('dlsdk_launcher', fields=DLSDKLauncher.parameters())

    def test_hetero_correct(self):
        self.config.validate(update_dict(self.launcher, device='HETERO:CPU'))
        self.config.validate(update_dict(self.launcher, device='HETERO:CPU,CPU'))

    def test_hetero_endswith_comma(self):
        with pytest.raises(ConfigError):
            self.config.validate(update_dict(self.launcher, device='HETERO:CPU,CPU,'))

    def test_normal_multiple_devices(self):
        with pytest.raises(ConfigError):
            self.config.validate(update_dict(self.launcher, device='CPU,CPU'))

    def test_hetero_empty(self):
        with pytest.raises(ConfigError):
            self.config.validate(update_dict(self.launcher, device='HETERO:'))

    def test_normal(self):
        self.config.validate(update_dict(self.launcher, device='CPU'))

    def test_missed_model_in_create_dlsdk_launcher_raises_config_error_exception(self):
        config = {'framework': 'dlsdk', 'weights': 'custom', 'adapter': 'classification', 'device': 'cpu'}

        with pytest.raises(ConfigError):
            create_launcher(config)

    def test_missed_adapter_in_create_dlsdk_launcher_raises_config_error_exception(self):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom'}

        with pytest.raises(ConfigError):
            create_launcher(launcher_config)

    def test_undefined_str_adapter_in_create_dlsdk_launcher_raises_config_error_exception(self):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': 'undefined_str'}

        with pytest.raises(ConfigError):
            create_launcher(launcher_config)

    def test_empty_dir_adapter_in_create_dlsdk_launcher_raises_config_error_exception(self):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': {}}

        with pytest.raises(ConfigError):
            create_launcher(launcher_config)

    def test_missed_type_in_dir_adapter_in_create_dlsdk_launcher_raises_config_error_exception(self):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': {'key': 'val'}}

        with pytest.raises(ConfigError):
            create_launcher(launcher_config)

    def test_undefined_type_in_dir_adapter_in_create_dlsdk_launcher_raises_config_error_exception(self):
        launcher_config = {
            'framework': 'dlsdk',
            'model': 'custom',
            'weights': 'custom',
            'adapter': {'type': 'undefined'}
        }

        with pytest.raises(ConfigError):
            create_launcher(launcher_config)

    def test_dlsdk_launcher(self):
        launcher = {
            'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': 'ssd', 'device': 'cpu'
        }
        create_launcher(launcher, model_name='custom')

    def test_dlsdk_launcher_model_with_several_image_inputs_raise_value_error(self, mocker):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': {'key': 'val'}}

        with pytest.raises(ValueError):
            mocker.patch(
                'accuracy_checker.launcher.dlsdk_launcher.DLSDKLauncher.inputs',
                new_callable=PropertyMock(return_value={'data1': [3, 227, 227], 'data2': [3, 227, 227]})
            )
            create_launcher(launcher_config)

    def test_dlsdk_launcher_model_no_image_inputs_raise_value_error(self):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': {'key': 'val'}}

        with pytest.raises(ValueError):
            create_launcher(launcher_config)
