"""
Copyright (c) 2018 Intel Corporation

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

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock
from accuracy_checker.config import ConfigError
from accuracy_checker.launcher import DLSDKLauncher
from accuracy_checker.launcher.dlsdk_launcher import DLSDKLauncherConfig
from accuracy_checker.launcher.launcher import create_launcher
from tests.common import update_dict
from accuracy_checker.dataset import DataRepresentation


@pytest.fixture()
def mock_inference_engine(mocker):
    try:
        mocker.patch('openvino.inference_engine.IEPlugin')
        mocker.patch('openvino.inference_engine.IENetwork')
    except ImportError:
        mocker.patch('inference_engine.IEPlugin')
        mocker.patch('inference_engine.IENetwork')


@pytest.fixture()
def mock_inputs(mocker):
    mocker.patch('accuracy_checker.launcher.dlsdk_launcher.DLSDKLauncher.inputs',
                 new_callable=PropertyMock(return_value={'data': [3, 227, 227]}))

@pytest.fixture()
def dlsdk_test_model(models_dir):
    config = {
        "framework": "dlsdk",
        "weights": str(models_dir / "SampLeNet.bin"),
        "model": str(models_dir / "SampLeNet.xml"),
        "device": "CPU",
        "adapter": 'classification'
    }

    model = create_launcher(config)
    return model


def test_infer(dlsdk_test_model, data_dir):
    c, h, w = dlsdk_test_model.inputs['data']
    img_raw = cv2.imread(str(data_dir / '1.jpg'))
    img_resized = cv2.resize(img_raw, (w, h))
    res = dlsdk_test_model.predict(['1.jpg'], [DataRepresentation(img_resized)])

    assert res[0].label == 6


def test_launcher_creates(dlsdk_test_model):
    assert dlsdk_test_model.inputs['data'] == [3, 32, 32]


@pytest.mark.usefixtures('mock_path_exists', 'mock_inference_engine', 'mock_inputs')
class TestDLSDKLauncher:
    def test_sets_dla_aocx_when_device_is_fpga(self, mocker):
        mocker.patch('os.environ')

        config = {
            "framework": "dlsdk",
            "weights": 'custom_weights',
            "model": 'custom_model',
            "device": 'fpga',
            "bitstream": "custom_bitstream",
            "adapter": 'classification'
        }
        create_launcher(config, {'label_map': {}})

        os.environ.__setitem__.assert_called_once_with('DLA_AOCX', 'custom_bitstream')

    def test_sets_dla_aocx_when_fpga_in_hetero_device(self, mocker):
        mocker.patch('os.environ')

        config = {
            "framework": "dlsdk",
            "weights": 'custom_weights',
            "model": 'custom_model',
            "device": 'hetero:fpga,cpu',
            "bitstream": "custom_bitstream",
            "adapter": 'classification'
        }
        create_launcher(config, {'label_map': {}})
        os.environ.__setitem__.assert_called_once_with('DLA_AOCX', 'custom_bitstream')

    def test_does_not_set_dla_aocx_when_device_is_not_fpga(self, mocker):
        mocker.patch('os.environ')

        config = {
            "framework": "dlsdk",
            "weights": 'custom_weights',
            "model": 'custom_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream",
            "adapter": "classification"
        }
        create_launcher(config)

        os.environ.__setitem__.assert_not_called()

    def test_does_not_set_dla_aocx_when_hetero_without_fpga(self, mocker):
        mocker.patch('os.environ')

        config = {
            "framework": "dlsdk",
            "weights": 'custom_weights',
            "model": 'custom_model',
            "device": 'hetero:cpu,cpu',
            "bitstream": "custom_bitstream",
            "adapter": "classification"
        }
        create_launcher(config)

        os.environ.__setitem__.assert_not_called()

    def test_does_not_set_dla_aocx_if_compiler_mode_3_in_env_when_fpga_in_hetero_device(self, mocker):
        mocker.patch('os.environ')
        mocker.patch('os.environ.get', return_value='3')

        config = {
            "framework": "dlsdk",
            "weights": 'custom_weights',
            "model": 'custom_model',
            "device": 'hetero:fpga,cpu',
            "bitstream": "custom_bitstream",
            "adapter": "classification"
        }
        create_launcher(config)

        os.environ.__setitem__.assert_not_called()

    def test_does_not_set_dla_aocx_if_compiler_mode_3_in_env_when_fpga_in_device(self, mocker):
        mocker.patch('os.environ')
        mocker.patch('os.environ.get', return_value='3')

        config = {
            "framework": "dlsdk",
            "weights": 'custom_weights',
            "model": 'custom_model',
            "device": 'fpga',
            "bitstream": "custom_bitstream",
            "adapter": "classification"
        }
        create_launcher(config)

        os.environ.__setitem__.assert_not_called()

    def test_model_converted_from_caffe(self, mocker):
        mock = mocker.patch('accuracy_checker.launcher.dlsdk_launcher.convert_model',
                            return_value=('converted_model', 'converted_weights'))  # type: MagicMock

        config = {
            "framework": "dlsdk",
            "caffe_model": '/path/to/source_models/custom_model',
            "caffe_weights": '/path/to/source_models/custom_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream",
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to/source_models"),
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        DLSDKLauncher(config, dummy_adapter)

        mock.assert_called_once_with(
            'custom_model', Path("/path/to/converted_models"), Path("/path/to/source_models/custom_model"),
            Path("/path/to/source_models/custom_weights"), 'caffe', [], {}, [], None
        )

    def test_model_converted_with_mo_params(self, mocker):
        mock = mocker.patch('accuracy_checker.launcher.dlsdk_launcher.convert_model',
                            return_value=('converted_model', 'converted_weights'))  # type: MagicMock

        config = {
            "framework": "dlsdk",
            "caffe_model": '/path/to/source_models/custom_model',
            "caffe_weights": '/path/to/source_models/custom_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream",
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to/source_models"),
            "mo_params": {'data_type': 'FP16'},
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        DLSDKLauncher(config, dummy_adapter)

        mock.assert_called_once_with(
            'custom_model', Path("/path/to/converted_models"), Path("/path/to/source_models/custom_model"),
            Path("/path/to/source_models/custom_weights"), 'caffe', [], {'data_type': 'FP16'}, [], None
        )

    def test_model_converted_with_mo_flags(self, mocker):
        mock = mocker.patch('accuracy_checker.launcher.dlsdk_launcher.convert_model',
                            return_value=('converted_model', 'converted_weights'))  # type: MagicMock

        config = {
            "framework": "dlsdk",
            "caffe_model": Path('/path/to/source_models/custom_model'),
            "caffe_weights": Path('/path/to/source_models/custom_weights'),
            "device": 'cpu',
            "bitstream": "custom_bitstream",
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to/source_models"),
            "mo_flags": ['reverse_input_channels'],
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        DLSDKLauncher(config, dummy_adapter)

        mock.assert_called_once_with(
            'custom_model', Path("/path/to/converted_models"), Path("/path/to/source_models/custom_model"),
            Path("/path/to/source_models/custom_weights"), 'caffe', [], {}, ['reverse_input_channels'], None
        )

    def test_model_converted_to_the_source_model_dir(self, mocker):
        config = {
            "framework": "dlsdk",
            "tf_model": Path('/path/to/source_models/custom_model'),
            "device": 'cpu',
            "_converted_models": Path("/path/to/source_models"),
            "_models_prefix": Path("/path/to/source_models"),
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value='ModelOptimizer')
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            '--input_model': Path('/path/to/source_models/custom_model'),
            '--model_name': 'custom_model',
            '--output_dir': Path('/path/to/source_models'),
            '--framework': 'tf'
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config, dummy_adapter)
        prepare_args_patch.assert_called_once_with('ModelOptimizer', flag_options=[], value_options=args)

    def test_model_converted_to_the_source_model_dir_with_suffix(self, mocker):
        config = {
            "framework": "dlsdk",
            "tf_model": Path('/path/to/source_models/custom_model'),
            "device": 'cpu',
            "_converted_models": Path("/path/to"),
            "_models_prefix": Path("/path/to"),
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value='ModelOptimizer')
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            '--input_model': Path('/path/to/source_models/custom_model'),
            '--model_name': 'custom_model',
            '--output_dir': Path('/path/to/source_models'),
            '--framework': 'tf'
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config, dummy_adapter)
        prepare_args_patch.assert_called_once_with('ModelOptimizer', flag_options=[], value_options=args)

    def test_model_converted_to_the_source_model_dir_with_postfix(self, mocker):
        config = {
            "framework": "dlsdk",
            "tf_model": Path('/path/to/source_models/custom_model'),
            "device": 'cpu',
            'converted_model_dir': 'models',
            "_converted_models": Path("/path/to/"),
            "_models_prefix": Path("/path/to"),
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value='ModelOptimizer')
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            '--input_model': Path('/path/to/source_models/custom_model'),
            '--model_name': 'custom_model',
            '--output_dir': Path('/path/to/models'),
            '--framework': 'tf'
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config, dummy_adapter)
        prepare_args_patch.assert_called_once_with('ModelOptimizer', flag_options=[], value_options=args)

    def test_model_converted_to_the_custom_dir(self, mocker):
        config = {
            "framework": "dlsdk",
            "tf_model": Path('/path/to/source_models/custom_model'),
            "device": 'cpu',
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to/source_models"),
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value='ModelOptimizer')
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            '--input_model': Path('/path/to/source_models/custom_model'),
            '--model_name': 'custom_model',
            '--output_dir': Path('/path/to/converted_models'),
            '--framework': 'tf'
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config, dummy_adapter)
        prepare_args_patch.assert_called_once_with('ModelOptimizer', flag_options=[], value_options=args)

    def test_model_converted_to_the_custom_dir_save_suffix(self, mocker):
        config = {
            "framework": "dlsdk",
            "tf_model": '/path/to/source_models/custom_model',
            "device": 'cpu',
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to"),
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value='ModelOptimizer')
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            '--input_model': Path('/path/to/source_models/custom_model'),
            '--model_name': 'custom_model',
            '--output_dir': Path('/path/to/converted_models/source_models/'),
            '--framework': 'tf'
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config, dummy_adapter)
        prepare_args_patch.assert_called_once_with('ModelOptimizer', flag_options=[], value_options=args)

    def test_model_converted_to_the_custom_dir_with_postfix(self, mocker):
        config = {
            "framework": "dlsdk",
            "tf_model": '/path/to/source_models/custom_model',
            "device": 'cpu',
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to"),
            "adapter": "classification",
            'converted_model_dir': 'models',
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value='ModelOptimizer')
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')

        args = {
            '--input_model': Path('/path/to/source_models/custom_model'),
            '--model_name': 'custom_model',
            '--output_dir': Path('/path/to/converted_models/models'),
            '--framework': 'tf'
        }

        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config, dummy_adapter)
        prepare_args_patch.assert_called_once_with('ModelOptimizer', flag_options=[], value_options=args)

    def test_model_conversion_to_output_dir_in_mo_params(self, mocker):
        config = {
            "framework": "dlsdk",
            "tf_model": '/path/to/source_models/custom_model',
            "device": 'cpu',
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to"),
            "adapter": "classification",
            'mo_params': {'output_dir': Path('/path/to/output/models')}
        }
        mocker.patch('accuracy_checker.launcher.model_conversion.find_mo', return_value='ModelOptimizer')
        prepare_args_patch = mocker.patch('accuracy_checker.launcher.model_conversion.prepare_args')
        args = {
            '--input_model': Path('/path/to/source_models/custom_model'),
            '--model_name': 'custom_model',
            '--output_dir': Path('/path/to/output/models'),
            '--framework': 'tf'
        }

        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        mocker.patch(
            'accuracy_checker.launcher.model_conversion.exec_mo_binary',
            return_value=subprocess.CompletedProcess(args, returncode=0)
        )
        DLSDKLauncher(config, dummy_adapter)
        prepare_args_patch.assert_called_once_with('ModelOptimizer', flag_options=[], value_options=args)

    def test_model_conversion_raises_config_error_if_output_dir_in_mo_params_and_converted_model_dir_both_provided(
            self, mocker
    ):
        config = {
            "framework": "dlsdk",
            "tf_model": '/path/to/source_models/custom_model',
            "device": 'cpu',
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to"),
            "adapter": "classification",
            'converted_model_dir': 'models',
            'mo_params': {'output_dir': Path('/path/to/output/models')}
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))

        with pytest.raises(ConfigError):
            DLSDKLauncher(config, dummy_adapter)

    def test_model_converted_from_tf(self, mocker):
        mock = mocker.patch('accuracy_checker.launcher.dlsdk_launcher.convert_model',
                            return_value=('converted_model', 'converted_weights'))  # type: MagicMock

        config = {
            "framework": "dlsdk",
            "tf_model": '/path/to/source_models/custom_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream",
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to/source_models"),
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        DLSDKLauncher(config, dummy_adapter)

        mock.assert_called_once_with(
            'custom_model', Path("/path/to/converted_models"), Path("/path/to/source_models/custom_model"),
            None, 'tf', [], {}, [], None
        )

    def test_model_converted_from_mxnet(self, mocker):
        mock = mocker.patch('accuracy_checker.launcher.dlsdk_launcher.convert_model',
                            return_value=('converted_model', 'converted_weights'))  # type: MagicMock

        config = {
            "framework": "dlsdk",
            "mxnet_weights": '/path/to/source_models/custom_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream",
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to/source_models"),
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        DLSDKLauncher(config, dummy_adapter)

        mock.assert_called_once_with(
            'custom_weights', Path("/path/to/converted_models"), None,
            Path("/path/to/source_models/custom_weights"), 'mxnet', [], {}, [], None
        )

    def test_model_converted_from_onnx(self, mocker):
        mock = mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.convert_model',
            return_value=('converted_model', 'converted_weights')
        )  # type: MagicMock

        config = {
            "framework": "dlsdk",
            "onnx_model": '/path/to/source_models/custom_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream",
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to/source_models"),
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        DLSDKLauncher(config, dummy_adapter)

        mock.assert_called_once_with(
            'custom_model', Path("/path/to/converted_models"), Path("/path/to/source_models/custom_model"),
            None, 'onnx', [], {}, [], None
        )

    def test_model_converted_from_kaldi(self, mocker):
        mock = mocker.patch(
            'accuracy_checker.launcher.dlsdk_launcher.convert_model',
            return_value=('converted_model', 'converted_weights')
        )  # type: MagicMock

        config = {
            "framework": "dlsdk",
            "kaldi_model": '/path/to/source_models/custom_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream",
            "_converted_models": Path("/path/to/converted_models"),
            "_models_prefix": Path("/path/to/source_models"),
            "adapter": "classification"
        }
        mocker.patch('pathlib.Path.resolve', return_value=Path(config['_converted_models']))
        DLSDKLauncher(config, dummy_adapter)

        mock.assert_called_once_with(
            'custom_model', Path("/path/to/converted_models"), Path("/path/to/source_models/custom_model"),
            None, 'kaldi', [], {}, [], None
        )

    def test_raises_with_multiple_models_caffe_dlsdk(self):
        config = {
            "framework": "dlsdk",
            "caffe_model": 'caffe_model',
            "caffe_weights": 'caffe_weights',
            "model": 'custom_model',
            "weights": 'custom_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_tf_dlsdk(self):
        config = {
            "framework": "dlsdk",
            "tf_model": 'tf_model',
            "model": 'custom_model',
            "weights": 'custom_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ConfigError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_mxnet_dlsdk(self):
        config = {
            "framework": "dlsdk",
            "mxnet_weights": 'mxnet_weights',
            "model": 'custom_model',
            "weights": 'custom_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_onnx_dlsdk(self):
        config = {
            "framework": "dlsdk",
            "onnx_model": 'onnx_model',
            "model": 'custom_model',
            "weights": 'custom_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_kaldi_dlsdk(self):
        config = {
            "framework": "dlsdk",
            "onnx_model": 'kaldi_model',
            "model": 'custom_model',
            "weights": 'custom_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_mxnet_caffe(self):
        config = {
            "framework": "dlsdk",
            "mxnet_weights": 'mxnet_weights',
            "caffe_model": 'caffe_model',
            "caffe_weights": 'caffe_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_tf_caffe(self, mocker):

        config = {
            "framework": "dlsdk",
            "tf_model": 'tf_model',
            "caffe_model": 'caffe_model',
            "caffe_weights": 'caffe_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_onnx_caffe(self, mocker):

        config = {
            "framework": "dlsdk",
            "onnx_model": 'onnx_model',
            "caffe_model": 'caffe_model',
            "caffe_weights": 'caffe_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_mxnet_tf(self):
        config = {
            "framework": "dlsdk",
            "mxnet_weights": 'mxnet_weights',
            "tf_model": 'tf_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_onnx_tf(self):
        config = {
            "framework": "dlsdk",
            "onnx_model": 'onnx_model',
            "tf_model": 'tf_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_mxnet_caffe_tf(self):
        config = {
            "framework": "dlsdk",
            "mxnet_weights": 'mxnet_weights',
            "caffe_model": 'caffe_model',
            "caffe_weights": 'caffe_weights',
            "tf_model": 'tf_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_dlsdk_caffe_tf(self):
        config = {
            "framework": "dlsdk",
            "model": 'custom_model',
            "weights": 'custom_weights',
            "caffe_model": 'caffe_model',
            "caffe_weights": 'caffe_weights',
            "tf_model": 'tf_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_dlsdk_caffe_onnx(self):
        config = {
            "framework": "dlsdk",
            "model": 'custom_model',
            "weights": 'custom_weights',
            "caffe_model": 'caffe_model',
            "caffe_weights": 'caffe_weights',
            "onnx_model": 'onnx_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_dlsdk_caffe_mxnet(self):
        config = {
            "framework": "dlsdk",
            "model": 'custom_model',
            "weights": 'custom_weights',
            "caffe_model": 'caffe_model',
            "caffe_weights": 'caffe_weights',
            "mxnet_weights": 'mxnet_weights',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_dlsdk_tf_mxnet(self):
        config = {
            "framework": "dlsdk",
            "model": 'custom_model',
            "weights": 'custom_weights',
            "mxnet_weights": 'mxnet_weights',
            "tf_model": 'tf_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_dlsdk_tf_onnx(self):
        config = {
            "framework": "dlsdk",
            "model": 'custom_model',
            "weights": 'custom_weights',
            "onnx_model": 'onnx_model',
            "tf_model": 'tf_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_dlsdk_tf_mxnet_caffe(self):
        config = {
            "framework": "dlsdk",
            "model": 'custom_model',
            "weights": 'custom_weights',
            "caffe_model": 'caffe_model',
            "caffe_weights": 'caffe_weights',
            "mxnet_weights": 'mxnet_weights',
            "onnx_model": 'onnx_model',
            "tf_model": 'tf_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }
        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)

    def test_raises_with_multiple_models_dlsdk_tf_mxnet_caffe_onnx(self):
        config = {
            "framework": "dlsdk",
            "model": 'custom_model',
            "weights": 'custom_weights',
            "caffe_model": 'caffe_model',
            "caffe_weights": 'caffe_weights',
            "mxnet_weights": 'mxnet_weights',
            "tf_model": 'tf_model',
            "device": 'cpu',
            "bitstream": "custom_bitstream"
        }

        with pytest.raises(ValueError):
            DLSDKLauncher(config, dummy_adapter)


@pytest.mark.usefixtures('mock_path_exists', 'mock_inputs', 'mock_inference_engine')
class TestDLSDKLauncherConfig:
    def setup(self):
        self.launcher = {
            "model": "foo.xml", "weights": "foo.bin",
            "device": "CPU", "framework": "dlsdk", "adapter": "classification"
        }
        self.config = DLSDKLauncherConfig('dlsdk_launcher')

    def test_hetero__correct(self):
        self.config.validate(update_dict(self.launcher, device="HETERO:CPU"))
        self.config.validate(update_dict(self.launcher, device="HETERO:CPU,FPGA"))

    def test_hetero__endswith_comma(self):
        with pytest.raises(ConfigError):
            self.config.validate(update_dict(self.launcher, device="HETERO:CPU,FPGA,"))

    def test_normal__multiple_devices(self):
        with pytest.raises(ConfigError):
            self.config.validate(update_dict(self.launcher, device="CPU,FPGA"))

    def test_hetero__empty(self):
        with pytest.raises(ConfigError):
            self.config.validate(update_dict(self.launcher, device="HETERO:"))

    def test_normal(self):
        self.config.validate(update_dict(self.launcher, device="CPU"))

    def test_missed_model_in_create_dlsdk_launcher_raises_config_error_exception(self):
        config = {'framework': 'dlsdk', 'weights': 'custom', 'adapter': 'classification', 'device': 'cpu'}

        with pytest.raises(ConfigError):
            create_launcher(config)

    def test_missed_weights_in_create_dlsdk_launcher_raises_config_error_exception(self):
        launcher = {'framework': 'dlsdk', 'model': 'custom', 'adapter': 'ssd', 'device': 'cpu'}

        with pytest.raises(ConfigError):
            create_launcher(launcher)

    def test_missed_adapter_in_create_dlsdk_launcher_raises_config_error_exception(self, mocker):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom'}

        with pytest.raises(ConfigError):
            create_launcher(launcher_config)

    def test_undefined_str_adapter_in_create_dlsdk_launcher_raises_config_error_exception(self, mocker):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': 'undefined_str'}

        with pytest.raises(ConfigError):
            create_launcher(launcher_config)

    def test_empty_dir_adapter_in_create_dlsdk_launcher_raises_config_error_exception(self, mocker):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': {}}

        with pytest.raises(ConfigError):
            create_launcher(launcher_config)

    def test_missed_type_in_dir_adapter_in_create_dlsdk_launcher_raises_config_error_exception(self, mocker):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': {'key': 'val'}}

        with pytest.raises(ConfigError):
            create_launcher(launcher_config)

    def test_undefined_type_in_dir_adapter_in_create_dlsdk_launcher_raises_config_error_exception(self, mocker):
        launcher_config = {
            'framework': 'dlsdk',
            'model': 'custom',
            'weights': 'custom',
            'adapter': {'type': 'undefined'}
        }

        with pytest.raises(ConfigError):
            create_launcher(launcher_config)

    def test_dlsdk_launcher(self, mocker):
        launcher = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': 'ssd', 'device': 'cpu'}
        create_launcher(launcher)

    def test_dlsdk_launcher_model_with_several_image_inputs_raise_value_error(self, mocker):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': {'key': 'val'}}

        with pytest.raises(ValueError):
            mocker.patch(
                'accuracy_checker.launcher.dlsdk_launcher.DLSDKLauncher.inputs',
                new_callable=PropertyMock(return_value={'data1': [3, 227, 227], 'data2': [3, 227, 227]})
            )
            create_launcher(launcher_config)

    def test_dlsdk_launcher_model_no_image_inputs_raise_value_error(self, mocker):
        launcher_config = {'framework': 'dlsdk', 'model': 'custom', 'weights': 'custom', 'adapter': {'key': 'val'}}

        with pytest.raises(ValueError):
            create_launcher(launcher_config)


def dummy_adapter():
    pass
