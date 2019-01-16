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

import copy
from pathlib import Path
from argparse import Namespace

import pytest
from accuracy_checker.config import ConfigReader


class TestConfigReader:
    def setup_method(self):
        self.global_launchers = [
            {
                'framework': 'dlsdk',
                'device': 'fpga',
                'cpu_extensions': 'dlsdk_shared.so',
                'bitstream': 'bitstream'
            },
            {
                'framework': 'caffe',
                'device': 'gpu_0'
            }
        ]

        self.global_datasets = [
            {
                'name': 'global_dataset',
                'annotation': 'annotations/pascal_voc_2007_annotation.pickle',
                'data_source': 'VOCdevkit/VOC2007/JPEGImages',
                'preprocessing': [
                    {
                        'type': 'resize',
                        'interpolation': 'mean_image',
                    },
                    {
                        'type': 'normalization',
                        'mean': 'voc',
                    }
                ],
                'metrics': [{
                    'type': 'fppi',
                    'mr_rates': [0.0, 0.1]
                }],
                'postprocessing': [
                    {
                        'type': 'filter',
                        'labels': ['dog', 'airplane'],
                        'min_confidence': 0.05,
                        'min_box_size': 60,
                    },
                    {
                        'type': 'nms',
                        'overlap': 0.5
                    }
                ]
            }
        ]

        self.global_config = {
            'launchers': self.global_launchers,
            'datasets': self.global_datasets
        }

        self.module = 'accuracy_checker.config.ConfigReader'
        self.arguments = Namespace(**{
            'models': Path('models'),
            'extensions': Path('extensions'),
            'source': Path('source'),
            'annotations': Path('annotations'),
            'converted_models': Path('converted_models'),
            'model_optimizer': Path('model_optimizer'),
            'bitstreams': Path('bitstreams'),
            'definitions': None,
            'stored_predictions': None,
            'tf_custom_op_config': None,
            'progress': 'bar',
            'target_framework': None,
            'target_devices': None,
            'log_file': None
        })

    def test_read_configs_without_global_config(self, mocker):
        config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': '/absolute_path', 'weights': '/absolute_path'}],
            'datasets': [{'name': 'global_dataset'}]
        }]}
        empty_args = Namespace(**{
            'models': None, 'extensions': None, 'source': None, 'annotations': None,
            'converted_models': None, 'model_optimizer': None, 'bitstreams': None,
            'definitions': None, 'config': None,'stored_predictions': None,'tf_custom_op_config': None,
            'progress': 'bar', 'target_framework': None, 'target_devices': None, 'log_file': None
        })
        mocker.patch('accuracy_checker.utils.get_path', return_value=Path.cwd())
        mocker.patch('yaml.load', return_value=config)
        mocker.patch('pathlib.Path.open')

        _, local_config = ConfigReader._read_configs(empty_args)

        assert config == local_config

    def test_empty_local_config_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {}
        ))

        with pytest.raises(ValueError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Missing local config'

    def test_missed_models_in_local_config_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'not_models': 'custom'}
        ))

        with pytest.raises(ValueError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Missed "{}" in local config'.format('models')

    def test_empty_models_in_local_config_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': []}
        ))

        with pytest.raises(ValueError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Missed "{}" in local config'.format('models')

    def test_missed_name_in_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'launchers': None, 'datasets': None}]}
        ))

        with pytest.raises(ValueError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(['name', 'launchers', 'datasets'])

    def test_missed_launchers_in_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'name': None, 'datasets': None}]}
        ))

        with pytest.raises(ValueError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(['name', 'launchers', 'datasets'])

    def test_missed_datasets_in_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'name': None, 'launchers': None}]}
        ))

        with pytest.raises(ValueError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(['name', 'launchers', 'datasets'])

    def test_invalid_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'name': None, 'launchers': None, 'datasets': None}]}
        ))

        with pytest.raises(ValueError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(['name', 'launchers', 'datasets'])

    def test_merge_datasets_with_definitions(self):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': '/absolute_path', 'weights': '/absolute_path'}],
            'datasets': [{'name': 'global_dataset'}]
        }]}

        config = ConfigReader._merge_configs(self.global_config, local_config)

        assert config['models'][0]['datasets'][0] == self.global_datasets[0]

    def test_merge_datasets_with_definitions_and_meta_is_not_modified(self):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': '/absolute_path', 'weights': '/absolute_path'}],
            'datasets': [{'name': 'global_dataset', 'dataset_meta': '/absolute_path'}]
        }]}
        expected = self.global_datasets[0]
        expected['dataset_meta'] = '/absolute_path'

        config = ConfigReader._merge_configs(self.global_config, local_config)

        assert config['models'][0]['datasets'][0] == expected

    def test_expand_relative_paths_in_datasets_config_using_command_line(self):
        local_config = {'models': [{
            'name': 'model',
            'datasets': [{
                'name': 'global_dataset',
                'dataset_meta': 'relative_annotation_path',
                'data_source': 'relative_source_path',
                'segmentation_masks_source': 'relative_source_path',
                'annotation': 'relative_annotation_path'
            }]
        }]}

        expected = copy.deepcopy(local_config['models'][0]['datasets'][0])
        expected['annotation'] = self.arguments.annotations / 'relative_annotation_path'
        expected['dataset_meta'] = self.arguments.annotations / 'relative_annotation_path'
        expected['segmentation_masks_source'] = self.arguments.source / 'relative_source_path'
        expected['data_source'] = self.arguments.source / 'relative_source_path'

        ConfigReader._merge_paths_with_prefixes(self.arguments, local_config)

        assert local_config['models'][0]['datasets'][0] == expected

    def test_merge_launchers_with_definitions(self):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk'}],
            'datasets': [{'name': 'global_dataset'}]
        }]}

        config = ConfigReader._merge_configs(self.global_config, local_config)

        assert config['models'][0]['launchers'][0] == self.get_global_launcher('dlsdk')

    def test_merge_launchers_with_model_is_not_modified(self):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': 'custom'}],
            'datasets': [{'name': 'global_dataset'}]
        }]}
        expected = self.get_global_launcher('dlsdk')
        expected['model'] = 'custom'

        config = ConfigReader._merge_configs(self.global_config, local_config)

        assert config['models'][0]['launchers'][0] == expected

    def test_expand_relative_paths_in_launchers_config_using_command_line(self):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{
                'framework': 'dlsdk',
                'model': 'relative_model_path',
                'weights': 'relative_weights_path',
                'cpu_extensions': 'relative_extensions_path',
                'gpu_extensions': 'relative_extensions_path',
                'caffe_model': 'relative_model_path',
                'caffe_weights': 'relative_weights_path',
                'tf_model': 'relative_model_path',
                'mxnet_weights': 'relative_weights_path',
                'bitstream': 'relative_bitstreams_path'
            }]
        }]}

        expected = copy.deepcopy(local_config['models'][0]['launchers'][0])
        expected['model'] = self.arguments.models / 'relative_model_path'
        expected['caffe_model'] = self.arguments.models / 'relative_model_path'
        expected['tf_model'] = self.arguments.models / 'relative_model_path'
        expected['weights'] = self.arguments.models / 'relative_weights_path'
        expected['caffe_weights'] = self.arguments.models / 'relative_weights_path'
        expected['mxnet_weights'] = self.arguments.models / 'relative_weights_path'
        expected['cpu_extensions'] = self.arguments.extensions / 'relative_extensions_path'
        expected['gpu_extensions'] = self.arguments.extensions / 'relative_extensions_path'
        expected['bitstream'] = self.arguments.bitstreams / 'relative_bitstreams_path'

        ConfigReader._merge_paths_with_prefixes(self.arguments, local_config)

        assert local_config['models'][0]['launchers'][0] == expected

    def test_launcher_is_not_filtered_by_the_same_framework(self):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': '/absolute_path1',
            'weights': '/absolute_path1',
            'adapter': 'classification',
            'device': 'CPU'
        }]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_framework = 'dlsdk'

        ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_both_launchers_are_not_filtered_by_the_same_framework(self):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU'
            },
            {
                'framework': 'dlsdk',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_framework = 'dlsdk'

        ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_launcher_is_filtered_by_another_framework(self):
        config = {'models': [{'name': 'name', 'launchers': [{
            'framework': 'dlsdk',
            'model': '/absolute_path',
            'weights': '/absolute_path',
            'adapter': 'classification'
        }]}]}
        self.arguments.target_framework = 'caffe'

        with pytest.warns(Warning):
            ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_both_launchers_are_filtered_by_another_framework(self):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU'
            },
            {
                'framework': 'dlsdk',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_framework = 'caffe'

        with pytest.warns(Warning):
            ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_only_appropriate_launcher_is_filtered_by_another_framework(self):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU'
            },
            {
                'framework': 'caffe',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_framework = 'caffe'

        ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[1]

    def test_launcher_is_not_filtered_by_the_same_device(self):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': '/absolute_path1',
            'weights': '/absolute_path1',
            'adapter': 'classification',
            'device': 'CPU'
        }]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_devices = ['CPU']

        ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_both_launchers_are_not_filtered_by_the_same_device(self):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU'
            },
            {
                'framework': 'caffe',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_devices = ['CPU']

        ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_launcher_is_filtered_by_another_device(self):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': '/absolute_path1',
            'weights': '/absolute_path1',
            'adapter': 'classification',
            'device': 'CPU'
        }]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_devices = ['GPU']

        with pytest.warns(Warning):
            ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_both_launchers_are_filtered_by_another_device(self):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU'
            },
            {
                'framework': 'caffe',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_devices = ['GPU']

        with pytest.warns(Warning):
            ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_only_appropriate_launcher_is_filtered_by_another_device(self):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU'
            },
            {
                'framework': 'caffe',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_devices = ['GPU']

        ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[1]


    def test_only_appropriate_launcher_is_filtered_by_user_input_devices(self):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU'
            },
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'HETERO:CPU,GPU'
            },
            {
                'framework': 'caffe',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_devices = ['GPU', 'CPU']

        ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert launchers == [config_launchers[0], config_launchers[2]]

    def test_both_launchers_are_filtered_by_other_devices(self):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU'
            },
            {
                'framework': 'caffe',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_devices = ['FPGA', 'MYRIAD']

        with pytest.warns(Warning):
            ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 0

    def test_both_launchers_are_not_filtered_by_same_devices(self):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU'
            },
            {
                'framework': 'caffe',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_devices = ['GPU', 'CPU']

        ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_launcher_is_not_filtered_by_device_with_tail(self):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': '/absolute_path1',
                'weights': '/absolute_path1',
                'adapter': 'classification',
                'device': 'CPU'
            },
            {
                'framework': 'caffe',
                'model': '/absolute_path2',
                'weights': '/absolute_path2',
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        config = {'models': [{'name': 'name', 'launchers': config_launchers}]}
        self.arguments.target_devices = ['CPU', 'GPUunexepectedtail']

        ConfigReader._filter_launchers(config, self.arguments)

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[0]

    def get_global_launcher(self, framework):
        for launcher in self.global_launchers:
            if launcher['framework'] == framework:
                return launcher

        raise ValueError('Undefined global launcher with framework = "{}"'.format(framework))

    def get_global_dataset(self, name):
        for dataset in self.global_datasets:
            if dataset['name'] == name:
                return dataset

        raise ValueError('Undefined global dataset with name = "{}"'.format(name))
