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

import subprocess

import pytest

pytest.importorskip('accuracy_checker.launcher.dlsdk_launcher')
import os
import cv2
import numpy as np

from pathlib import Path
from unittest.mock import PropertyMock
from accuracy_checker.config import ConfigError
from accuracy_checker.launcher import DLSDKLauncher
from accuracy_checker.launcher.dlsdk_launcher_config import DLSDKLauncherConfigValidator
from accuracy_checker.launcher.launcher import create_launcher
from accuracy_checker.launcher.model_conversion import FrameworkParameters
from tests.common import update_dict
from accuracy_checker.data_readers import DataRepresentation
from accuracy_checker.utils import contains_all


def no_available_myriad():
    try:
        from openvino.inference_engine import IECore
        return 'MYRIAD' not in IECore().availabe_devices
    except:
        return True


@pytest.fixture()
def mock_inference_engine(mocker):
    mocker.patch('openvino.inference_engine.IECore')
    mocker.patch('openvino.inference_engine.IENetwork')


@pytest.fixture()
def mock_inputs(mocker):
    mocker.patch(
        'accuracy_checker.launcher.input_feeder.InputFeeder._parse_inputs_config', return_value=({}, ['data'], None)
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

    return create_launcher(config)


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

    def test_dlsdk_laucher_model_search(self, models_dir):
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



class TestDLSDKLauncherAffinity:
    @pytest.mark.usefixtures('mock_affinity_map_exists')
    def test_dlsdk_launcher_valid_affinity_map(self, mocker, models_dir):
        affinity_map = {'conv1': 'GPU'}

        mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.read_yaml', return_value=affinity_map
        )

        dlsdk_test_model = get_dlsdk_test_model(models_dir, {
            'device': 'HETERO:CPU,GPU', 'affinity_map': './affinity_map.yml'
        })
        layers = dlsdk_test_model.network.layers
        for key, value in affinity_map.items():
            assert layers[key].affinity == value

    @pytest.mark.usefixtures('mock_file_exists')
    def test_dlsdk_launcher_affinity_map_invalid_device(self, mocker, models_dir):
        affinity_map = {'conv1': 'GPU'}

        mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.read_yaml', return_value=affinity_map
        )

        with pytest.raises(ConfigError):
            get_dlsdk_test_model(models_dir, {'device' : 'HETERO:CPU,CPU', 'affinity_map' : './affinity_map.yml'})

    @pytest.mark.usefixtures('mock_file_exists')
    def test_dlsdk_launcher_affinity_map_invalid_layer(self, mocker, models_dir):
        affinity_map = {'none-existing-layer' : 'CPU'}

        mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.read_yaml', return_value=affinity_map
        )

        with pytest.raises(ConfigError):
            get_dlsdk_test_model(models_dir, {'device' : 'HETERO:CPU,CPU', 'affinity_map' : './affinity_map.yml'})


@pytest.mark.usefixtures('mock_path_exists', 'mock_inference_engine', 'mock_inputs')
class TestDLSDKLauncher:
    FAKE_MO_PATH = Path('/path/ModelOptimizer').absolute()

    def test_program_bitsream_when_device_is_fpga(self, mocker):
        subprocess_mock = mocker.patch('subprocess.run')
        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'fpga',
            'bitstream': Path('custom_bitstream'),
            'adapter': 'classification',
            '_aocl': Path('aocl')
        }
        launcher = create_launcher(config)
        subprocess_mock.assert_called_once_with(['aocl', 'program', 'acl0', 'custom_bitstream'], check=True)
        launcher.release()

    def test_program_bitstream_when_fpga_in_hetero_device(self, mocker):
        subprocess_mock = mocker.patch('subprocess.run')
        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'hetero:fpga,cpu',
            'bitstream': Path('custom_bitstream'),
            'adapter': 'classification',
            '_aocl': Path('aocl')
        }
        launcher = create_launcher(config)
        subprocess_mock.assert_called_once_with(['aocl', 'program', 'acl0', 'custom_bitstream'], check=True)
        launcher.release()

    def test_does_not_program_bitstream_when_device_is_not_fpga(self, mocker):
        subprocess_mock = mocker.patch('subprocess.run')
        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'cpu',
            'bitstream': Path('custom_bitstream'),
            'adapter': 'classification',
            '_aocl': Path('aocl')
        }
        create_launcher(config)
        subprocess_mock.assert_not_called()

    def test_does_not_program_bitstream_when_hetero_without_fpga(self, mocker):
        subprocess_mock = mocker.patch('subprocess.run')

        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'hetero:cpu,cpu',
            'bitstream': Path('custom_bitstream'),
            'adapter': 'classification',
            '_aocl': Path('aocl')
        }
        create_launcher(config)
        subprocess_mock.assert_not_called()

    def test_does_not_program_bitstream_if_compiler_mode_3_in_env_when_fpga_in_hetero_device(self, mocker):
        subprocess_mock = mocker.patch('subprocess.run')
        mocker.patch('os.environ.get', return_value='3')

        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'hetero:fpga,cpu',
            'bitstream': Path('custom_bitstream'),
            'adapter': 'classification',
            '_aocl': Path('aocl')
        }
        create_launcher(config)

        subprocess_mock.assert_not_called()

    def test_does_not_program_bitstream_if_compiler_mode_3_in_env_when_fpga_in_device(self, mocker):
        subprocess_mock = mocker.patch('subprocess.run')
        mocker.patch('os.environ.get', return_value='3')

        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'fpga',
            'bitstream': Path('custom_bitstream'),
            'adapter': 'classification',
            '_aocl': Path('aocl')
        }
        create_launcher(config)

        subprocess_mock.assert_not_called()

    def test_sets_dla_aocx_when_device_is_fpga(self, mocker):
        mocker.patch('os.environ')

        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'fpga',
            'bitstream': Path('custom_bitstream'),
            'adapter': 'classification',
        }
        create_launcher(config)

        os.environ.__setitem__.assert_called_once_with('DLA_AOCX', 'custom_bitstream')

    def test_sets_dla_aocx_when_fpga_in_hetero_device(self, mocker):
        mocker.patch('os.environ')

        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'hetero:fpga,cpu',
            'bitstream': Path('custom_bitstream'),
            'adapter': 'classification',
        }
        create_launcher(config)
        os.environ.__setitem__.assert_called_once_with('DLA_AOCX', 'custom_bitstream')

    def test_does_not_set_dla_aocx_when_device_is_not_fpga(self, mocker):
        mocker.patch('os.environ')

        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'cpu',
            'bitstream': 'custom_bitstream',
            'adapter': 'classification',
        }
        create_launcher(config)

        os.environ.__setitem__.assert_not_called()

    def test_does_not_set_dla_aocx_when_hetero_without_fpga(self, mocker):
        mocker.patch('os.environ')

        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'hetero:cpu,cpu',
            'bitstream': 'custom_bitstream',
            'adapter': 'classification',
        }
        create_launcher(config)

        os.environ.__setitem__.assert_not_called()

    def test_does_not_set_dla_aocx_if_compiler_mode_3_in_env_when_fpga_in_hetero_device(self, mocker):
        mocker.patch('os.environ')
        mocker.patch('os.environ.get', return_value='3')

        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'hetero:fpga,cpu',
            'bitstream': 'custom_bitstream',
            'adapter': 'classification',
        }
        create_launcher(config)

        os.environ.__setitem__.assert_not_called()

    def test_does_not_set_dla_aocx_if_compiler_mode_3_in_env_when_fpga_in_device(self, mocker):
        mocker.patch('os.environ')
        mocker.patch('os.environ.get', return_value='3')

        config = {
            'framework': 'dlsdk',
            'weights': 'custom_weights',
            'model': 'custom_model',
            'device': 'fpga',
            'bitstream': 'custom_bitstream',
            'adapter': 'classification',
        }
        create_launcher(config)

        os.environ.__setitem__.assert_not_called()

    def test_model_converted_from_caffe(self, mocker):
        mock = mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.convert_model',
            return_value=('converted_model', 'converted_weights')
        )

        config = {
            'framework': 'dlsdk',
            'caffe_model': '/path/to/source_models/custom_model',
            'caffe_weights': '/path/to/source_models/custom_weights',
            "device": 'cpu',
            'bitstream': Path('custom_bitstream'),
            'adapter': 'classification',
            'should_log_cmd': False
        }
        DLSDKLauncher(config)

        mock.assert_called_once_with(
            'custom_model', '/path/to/source_models/custom_model', '/path/to/source_models/custom_weights', '',
            FrameworkParameters('caffe', False),
            [], None, None, None, None, None, should_log_cmd=False
        )

    def test_model_converted_with_mo_params(self, mocker):
        mock = mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.convert_model',
            return_value=('converted_model', 'converted_weights')
        )

        config = {
            'framework': "dlsdk",
            'caffe_model': '/path/to/source_models/custom_model',
            'caffe_weights': '/path/to/source_models/custom_weights',
            'device': 'cpu',
            'bitstream': Path('custom_bitstream'),
            'mo_params': {'data_type': 'FP16'},
            'adapter': 'classification',
            'should_log_cmd': False
        }
        DLSDKLauncher(config)

        mock.assert_called_once_with(
            'custom_model', '/path/to/source_models/custom_model', '/path/to/source_models/custom_weights', '',
            FrameworkParameters('caffe', False),
            [], {'data_type': 'FP16'}, None, None, None, None, should_log_cmd=False
        )

    def test_model_converted_with_mo_flags(self, mocker):
        mock = mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.convert_model',
            return_value=('converted_model', 'converted_weights')
        )

        config = {
            'framework': 'dlsdk',
            'caffe_model': '/path/to/source_models/custom_model',
            'caffe_weights': '/path/to/source_models/custom_weights',
            'device': 'cpu',
            'bitstream': Path('custom_bitstream'),
            'mo_flags': ['reverse_input_channels'],
            'adapter': 'classification',
            'should_log_cmd': False
        }

        DLSDKLauncher(config)

        mock.assert_called_once_with(
            'custom_model', '/path/to/source_models/custom_model', '/path/to/source_models/custom_weights', '',
            FrameworkParameters('caffe', False),
            [], None, ['reverse_input_channels'], None, None, None, should_log_cmd=False
        )

    def test_model_converted_to_output_dir_in_mo_params(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_model': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'output_dir': '/path/to/output/models'}
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value='ModelOptimizer')
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')
        args = {
            'input_model': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'output_dir': '/path/to/output/models',
            'framework': 'tf'
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with('ModelOptimizer', flag_options=[], value_options=args)

    def test_model_converted_from_tf(self, mocker):
        mock = mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.convert_model',
            return_value=('converted_model', 'converted_weights')
        )

        config = {
            'framework': 'dlsdk',
            'tf_model': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'should_log_cmd': False
        }
        DLSDKLauncher(config)

        mock.assert_called_once_with(
            'custom_model', '/path/to/source_models/custom_model', '', '',
            FrameworkParameters('tf', False), [], None, None, None, None, None,
            should_log_cmd=False
        )

    def test_model_converted_from_tf_checkpoint(self, mocker):
        mock = mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.convert_model',
            return_value=('converted_model', 'converted_weights')
        )

        config = {
            'framework': 'dlsdk',
            'tf_meta': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'should_log_cmd': False
        }
        DLSDKLauncher(config)

        mock.assert_called_once_with(
            'custom_model', '', '', '/path/to/source_models/custom_model',
            FrameworkParameters('tf', True), [], None, None, None, None, None,
            should_log_cmd=False
        )

    def test_model_converted_from_tf_with_arg_path_to_custom_tf_config(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_model': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'tensorflow_use_custom_operations_config': 'ssd_v2_support.json'},
            '_tf_custom_op_config_dir': 'config/dir'
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value=self.FAKE_MO_PATH)
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            'input_model': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'framework': 'tf',
            'tensorflow_use_custom_operations_config': str(Path('config/dir/ssd_v2_support.json'))
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        mo_path = str(self.FAKE_MO_PATH)
        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with(mo_path, flag_options=[], value_options=args)

    def test_model_converted_from_tf_with_default_path_to_custom_tf_config(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_model': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'tensorflow_use_custom_operations_config': 'config.json'}
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value=self.FAKE_MO_PATH)
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            'input_model': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'framework': 'tf',
            'tensorflow_use_custom_operations_config': str(Path('/path/extensions/front/tf/config.json').absolute())
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with(str(self.FAKE_MO_PATH), flag_options=[], value_options=args)

    def test_model_converted_from_tf_with_default_path_to_obj_detection_api_config(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_model': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'tensorflow_object_detection_api_pipeline_config': 'operations.config'},
            '_tf_obj_detection_api_pipeline_config_path': None
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value=self.FAKE_MO_PATH)
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            'input_model': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'framework': 'tf',
            'tensorflow_object_detection_api_pipeline_config': str(Path('/path/to/source_models/operations.config'))
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with(str(self.FAKE_MO_PATH), flag_options=[], value_options=args)

    def test_model_converted_from_tf_with_arg_path_to_obj_detection_api_config(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_model': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'tensorflow_object_detection_api_pipeline_config': 'operations.config'},
            '_tf_obj_detection_api_pipeline_config_path': 'od_api'
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value=self.FAKE_MO_PATH)
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            'input_model': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'framework': 'tf',
            'tensorflow_object_detection_api_pipeline_config': str(Path('od_api/operations.config'))
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with(str(self.FAKE_MO_PATH), flag_options=[], value_options=args)

    def test_model_converted_from_tf_checkpoint_with_arg_path_to_custom_tf_config(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_meta': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'tensorflow_use_custom_operations_config': 'ssd_v2_support.json'},
            '_tf_custom_op_config_dir': 'config/dir'
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value=self.FAKE_MO_PATH)
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            'input_meta_graph': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'framework': 'tf',
            'tensorflow_use_custom_operations_config': str(Path('config/dir/ssd_v2_support.json'))
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with(str(self.FAKE_MO_PATH), flag_options=[], value_options=args)

    def test_model_converted_from_tf_checkpoint_with_default_path_to_custom_tf_config(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_meta': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'tensorflow_use_custom_operations_config': 'config.json'}
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value=self.FAKE_MO_PATH)
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            'input_meta_graph': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'framework': 'tf',
            'tensorflow_use_custom_operations_config': str(Path('/path/extensions/front/tf/config.json').absolute())
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with(str(self.FAKE_MO_PATH), flag_options=[], value_options=args)

    def test_model_converted_from_tf_checkpoint_with_default_path_to_obj_detection_api_config(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_meta': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'tensorflow_object_detection_api_pipeline_config': 'operations.config'},
            '_tf_obj_detection_api_pipeline_config_path': None
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value=self.FAKE_MO_PATH)
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            'input_meta_graph': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'framework': 'tf',
            'tensorflow_object_detection_api_pipeline_config': str(Path('/path/to/source_models/operations.config'))
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with(str(self.FAKE_MO_PATH), flag_options=[], value_options=args)

    def test_model_converted_from_tf_checkpoint_with_arg_path_to_obj_detection_api_config(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_meta': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'tensorflow_object_detection_api_pipeline_config': 'operations.config'},
            '_tf_custom_op_config_dir': 'config/dir',
            '_tf_obj_detection_api_pipeline_config_path': 'od_api'
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value=self.FAKE_MO_PATH)
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            'input_meta_graph': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'framework': 'tf',
            'tensorflow_object_detection_api_pipeline_config': str(Path('od_api/operations.config'))
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with(str(self.FAKE_MO_PATH), flag_options=[], value_options=args)

    def test_model_converted_from_tf_checkpoint_with_arg_path_to_transformations_config(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_meta': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'transformations_config': 'ssd_v2_support.json'},
            '_transformations_config_dir': 'config/dir'
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value=self.FAKE_MO_PATH)
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            'input_meta_graph': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'framework': 'tf',
            'transformations_config': str(Path('config/dir/ssd_v2_support.json'))
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with(str(self.FAKE_MO_PATH), flag_options=[], value_options=args)

    def test_model_converted_from_tf_checkpoint_with_default_path_to_transformations_config(self, mocker):
        config = {
            'framework': 'dlsdk',
            'tf_meta': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'mo_params': {'transformations_config': 'config.json'}
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value=self.FAKE_MO_PATH)
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            'input_meta_graph': '/path/to/source_models/custom_model',
            'model_name': 'custom_model',
            'framework': 'tf',
            'transformations_config': str(Path('/path/extensions/front/tf/config.json').absolute())
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )

        DLSDKLauncher(config)
        prepare_args_patch.assert_called_once_with(str(self.FAKE_MO_PATH), flag_options=[], value_options=args)

    def test_model_converted_from_mxnet(self, mocker):
        mock = mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.convert_model',
            return_value=('converted_model', 'converted_weights')
        )

        config = {
            'framework': 'dlsdk',
            'mxnet_weights': '/path/to/source_models/custom_weights',
            'device': 'cpu',
            'adapter': 'classification',
            'should_log_cmd': False
        }
        DLSDKLauncher(config)

        mock.assert_called_once_with(
            'custom_weights', '', '/path/to/source_models/custom_weights', '',
            FrameworkParameters('mxnet', False), [], None, None, None, None, None,
            should_log_cmd=False
        )

    def test_model_converted_from_onnx(self, mocker):
        mock = mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.convert_model',
            return_value=('converted_model', 'converted_weights')
        )

        config = {
            'framework': 'dlsdk',
            'onnx_model': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'should_log_cmd': False
        }
        DLSDKLauncher(config)

        mock.assert_called_once_with(
            'custom_model', '/path/to/source_models/custom_model', '', '',
            FrameworkParameters('onnx', False), [], None, None, None, None, None,
            should_log_cmd=False
        )

    def test_model_converted_from_kaldi(self, mocker):
        mock = mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.convert_model',
            return_value=('converted_model', 'converted_weights')
        )

        config = {
            'framework': 'dlsdk',
            'kaldi_model': '/path/to/source_models/custom_model',
            'device': 'cpu',
            'adapter': 'classification',
            'should_log_cmd': False
        }
        DLSDKLauncher(config)

        mock.assert_called_once_with(
            'custom_model', '/path/to/source_models/custom_model', '', '',
            FrameworkParameters('kaldi', False), [], None, None, None, None, None,
            should_log_cmd=False
        )

    def test_raises_with_multiple_models_caffe_dlsdk(self):
        config = {
            'framework': 'dlsdk',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_tf_dlsdk(self):
        config = {
            'framework': 'dlsdk',
            'tf_model': 'tf_model',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_mxnet_dlsdk(self):
        config = {
            'framework': 'dlsdk',
            'mxnet_weights': 'mxnet_weights',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_onnx_dlsdk(self):
        config = {
            'framework': 'dlsdk',
            'onnx_model': 'onnx_model',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_kaldi_dlsdk(self):
        config = {
            'framework': 'dlsdk',
            'onnx_model': 'kaldi_model',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_mxnet_caffe(self):
        config = {
            'framework': 'dlsdk',
            'mxnet_weights': 'mxnet_weights',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_tf_caffe(self):

        config = {
            'framework': 'dlsdk',
            'tf_model': 'tf_model',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_onnx_caffe(self):

        config = {
            'framework': 'dlsdk',
            'onnx_model': 'onnx_model',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_mxnet_tf(self):
        config = {
            'framework': 'dlsdk',
            'mxnet_weights': 'mxnet_weights',
            'tf_model': 'tf_model',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_onnx_tf(self):
        config = {
            'framework': 'dlsdk',
            'onnx_model': 'onnx_model',
            'tf_model': 'tf_model',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_mxnet_caffe_tf(self):
        config = {
            'framework': 'dlsdk',
            'mxnet_weights': 'mxnet_weights',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'tf_model': 'tf_model',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_dlsdk_caffe_tf(self):
        config = {
            'framework': 'dlsdk',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'tf_model': 'tf_model',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_dlsdk_caffe_onnx(self):
        config = {
            'framework': 'dlsdk',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'onnx_model': 'onnx_model',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_dlsdk_caffe_mxnet(self):
        config = {
            'framework': 'dlsdk',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'mxnet_weights': 'mxnet_weights',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_dlsdk_tf_mxnet(self):
        config = {
            'framework': "dlsdk",
            'model': 'custom_model',
            'weights': 'custom_weights',
            'mxnet_weights': 'mxnet_weights',
            'tf_model': 'tf_model',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_dlsdk_tf_onnx(self):
        config = {
            'framework': 'dlsdk',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'onnx_model': 'onnx_model',
            'tf_model': 'tf_model',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_dlsdk_tf_mxnet_caffe(self):
        config = {
            'framework': 'dlsdk',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'mxnet_weights': 'mxnet_weights',
            'onnx_model': 'onnx_model',
            'tf_model': 'tf_model',
            'device': 'cpu',
        }
        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_multiple_models_dlsdk_tf_mxnet_caffe_onnx(self):
        config = {
            'framework': 'dlsdk',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'mxnet_weights': 'mxnet_weights',
            'tf_model': 'tf_model',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    def test_raises_with_tf_model_and_tf_meta_both_provided(self):
        config = {
            'framework': 'dlsdk',
            'model': 'custom_model',
            'weights': 'custom_weights',
            'caffe_model': 'caffe_model',
            'caffe_weights': 'caffe_weights',
            'mxnet_weights': 'mxnet_weights',
            'tf_model': 'tf_model',
            'tf_meta': 'tf_meta',
            'device': 'cpu',
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config)

    @pytest.mark.usefixtures('mock_file_exists')
    def test_dlsdk_launcher_device_config_config_not_dict_like(self, mocker, models_dir):
        device_config = 'ENFORCE_BF16'

        mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.read_yaml', return_value=device_config
        )

        with pytest.raises(ConfigError):
            get_dlsdk_test_model(models_dir, {'_device_config': './device_config.yml'})

    @pytest.mark.usefixtures('mock_file_exists')
    def test_dlsdk_launcher_device_config_device_unknown(self, mocker, models_dir):
        device_config = {'device': {'ENFORCE_BF16': 'NO'}}

        mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.read_yaml', return_value=device_config
        )

        with pytest.warns(Warning):
            get_dlsdk_test_model(models_dir, {'_device_config': './device_config.yml'})

    @pytest.mark.usefixtures('mock_file_exists')
    def test_dlsdk_launcher_device_config_one_option_for_device_is_not_dict(self, mocker, models_dir):
        device_config = {'CPU': {'ENFORCE_BF16': 'NO'}, 'GPU': 'ENFORCE_BF16'}

        mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.read_yaml', return_value=device_config
        )

        with pytest.warns(Warning):
            get_dlsdk_test_model(models_dir, {'_device_config': './device_config.yml'})

    @pytest.mark.usefixtures('mock_file_exists')
    def test_dlsdk_launcher_device_config_one_option_is_not_binding_to_device(self, mocker, models_dir):
        device_config = {'CPU': {'ENFORCE_BF16': 'NO'}, 'ENFORCE_BF16': 'NO'}

        mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.read_yaml', return_value=device_config
        )

        with pytest.warns(Warning):
            get_dlsdk_test_model(models_dir, {'_device_config': './device_config.yml'})


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
        self.config.validate(update_dict(self.launcher, device='HETERO:CPU,FPGA'))

    def test_hetero_endswith_comma(self):
        with pytest.raises(ConfigError):
            self.config.validate(update_dict(self.launcher, device='HETERO:CPU,FPGA,'))

    def test_normal_multiple_devices(self):
        with pytest.raises(ConfigError):
            self.config.validate(update_dict(self.launcher, device='CPU,FPGA'))

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
        create_launcher(launcher)

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
