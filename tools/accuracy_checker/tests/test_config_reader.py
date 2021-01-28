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

import copy
from pathlib import Path
from argparse import Namespace

import pytest
from .common import mock_filesystem
from accuracy_checker.config import ConfigReader, ConfigError


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
                'annotation': Path('/pascal_voc_2007_annotation.pickle').absolute(),
                'data_source': Path('/VOCdevkit/VOC2007/JPEGImages').absolute(),
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
            'models': Path('models/'),
            'extensions': Path('extensions/'),
            'source': Path('source/'),
            'annotations': Path('annotations/'),
            'converted_models': Path('converted_models/'),
            'model_optimizer': Path('model_optimizer/'),
            'bitstreams': Path('bitstreams/'),
            'definitions': None,
            'stored_predictions': None,
            'tf_custom_op_config_dir': None,
            'tf_obj_detection_api_pipeline_config_path': None,
            'progress': 'bar',
            'target_framework': None,
            'target_devices': None,
            'log_file': None,
            'target_tags': None,
            'cpu_extensions_mode': None,
            'aocl': None,
            'deprecated_ir_v7': False,
            'transformations_config_dir': None,
            'model_attributes': None
        })

    def test_read_configs_without_global_config(self, mocker):
        config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': Path('/absolute_path').absolute(), 'weights': Path('/absolute_path').absolute()}],
            'datasets': [{'name': 'global_dataset'}]
        }]}
        empty_args = Namespace(**{
            'models': Path.cwd(), 'extensions': Path.cwd(), 'source': Path.cwd(), 'annotations': Path.cwd(),
            'converted_models': None, 'model_optimizer': None, 'bitstreams': Path.cwd(),
            'definitions': None, 'config': None, 'stored_predictions': None, 'tf_custom_op_config_dir': None,
            'progress': 'bar', 'target_framework': None, 'target_devices': None, 'log_file': None,
            'tf_obj_detection_api_pipeline_config_path': None, 'target_tags': None, 'cpu_extensions_mode': None,
            'aocl': None, 'deprecated_ir_v7': False, 'transformations_config_dir': None, 'model_attributes': None
        })
        mocker.patch('accuracy_checker.utils.get_path', return_value=Path.cwd())
        mocker.patch('yaml.load', return_value=config)
        mocker.patch('pathlib.Path.open')

        result = ConfigReader.merge(empty_args)

        assert 'models' == result[1]
        assert config == result[0]

    def test_empty_local_config_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception.value).split(sep=': ')[-1]
        assert error_message == 'Missing local config'

    def test_missed_models_in_local_config_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'not_models': 'custom'}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception.value).split(sep=': ')[-1]
        assert error_message == 'Accuracy Checker not_models mode is not supported. Please select between evaluations and models.'

    def test_empty_models_in_local_config_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': []}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception.value).split(sep=': ')[-1]
        assert error_message == 'Missed "{}" in local config'.format('models')

    def test_missed_name_in_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'launchers': None, 'datasets': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception.value).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(', '.join(['name', 'launchers', 'datasets']))

    def test_missed_launchers_in_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'name': None, 'datasets': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception.value).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(', '.join(['name', 'launchers', 'datasets']))

    def test_missed_datasets_in_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'name': None, 'launchers': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception.value).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(', '.join(['name', 'launchers', 'datasets']))

    def test_invalid_model_raises_value_error_exception(self, mocker):
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, {'models': [{'name': None, 'launchers': None, 'datasets': None}]}
        ))

        with pytest.raises(ConfigError) as exception:
            ConfigReader.merge(self.arguments)

        error_message = str(exception.value).split(sep=': ')[-1]
        assert error_message == 'Each model must specify {}'.format(', '.join(['name', 'launchers', 'datasets']))

    def test_merge_datasets_with_definitions(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': Path('/absolute_path').absolute(), 'weights': Path('/absolute_path').absolute()}],
            'datasets': [{'name': 'global_dataset'}]
        }]}
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, local_config
        ))
        arguments = copy.deepcopy(self.arguments)
        arguments.model_optimizer = None
        arguments.extensions = None
        arguments.bitstreams = None

        config = ConfigReader.merge(arguments)[0]

        assert config['models'][0]['datasets'][0] == self.global_datasets[0]

    def test_merge_datasets_with_definitions_and_meta_is_not_modified(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': Path('/absolute_path').absolute(), 'weights': Path('/absolute_path').absolute()}],
            'datasets': [{'name': 'global_dataset', 'dataset_meta': Path('/absolute_path').absolute()}]
        }]}
        expected = self.global_datasets[0]
        expected['dataset_meta'] = Path('/absolute_path').absolute()
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, local_config
        ))
        arguments = copy.deepcopy(self.arguments)
        arguments.bitstreams = None
        arguments.extensions = None

        config = ConfigReader.merge(arguments)[0]

        assert config['models'][0]['datasets'][0] == expected

    def test_expand_relative_paths_in_datasets_config_using_command_line(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'caffe'}],
            'datasets': [{
                'name': 'global_dataset',
                'dataset_meta': 'relative_annotation_path',
                'data_source': 'relative_source_path',
                'segmentation_masks_source': 'relative_source_path',
                'annotation': 'relative_annotation_path'
            }]
        }]}

        mocker.patch(self.module + '._read_configs', return_value=(
            None, local_config
        ))
        expected = copy.deepcopy(local_config['models'][0]['datasets'][0])
        with mock_filesystem(['source/', 'annotations/']) as prefix:
            expected['annotation'] = prefix / self.arguments.annotations / 'relative_annotation_path'
            expected['dataset_meta'] = prefix / self.arguments.annotations / 'relative_annotation_path'
            expected['segmentation_masks_source'] = prefix / self.arguments.source / 'relative_source_path'
            expected['data_source'] = prefix / self.arguments.source / 'relative_source_path'

            arguments = copy.deepcopy(self.arguments)
            arguments.bitstreams = None
            arguments.extensions = None
            arguments.annotations = prefix / self.arguments.annotations
            arguments.source = prefix / self.arguments.source

            config = ConfigReader.merge(arguments)[0]

            assert config['models'][0]['datasets'][0] == expected

    def test_expand_relative_paths_in_datasets_config_using_env_variable(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'caffe'}],
            'datasets': [{
                'name': 'global_dataset',
                'dataset_meta': 'relative_annotation_path',
                'data_source': 'relative_source_path',
                'segmentation_masks_source': 'relative_source_path',
                'annotation': 'relative_annotation_path'
            }]
        }]}

        mocker.patch(self.module + '._read_configs', return_value=(
            None, local_config
        ))
        expected = copy.deepcopy(local_config['models'][0]['datasets'][0])
        with mock_filesystem(['source_2/']) as env_prefix:
            mocker.patch('os.environ.get', return_value=str(env_prefix))
        with mock_filesystem(['source/', 'annotations/']) as prefix:
            expected['annotation'] = prefix / self.arguments.annotations / 'relative_annotation_path'
            expected['dataset_meta'] = prefix / self.arguments.annotations / 'relative_annotation_path'
            expected['segmentation_masks_source'] = prefix / self.arguments.source / 'relative_source_path'
            expected['data_source'] = prefix / self.arguments.source / 'relative_source_path'

            arguments = copy.deepcopy(self.arguments)
            arguments.bitstreams = None
            arguments.extensions = None
            arguments.source = prefix / arguments.source
            arguments.annotations = prefix / self.arguments.annotations

            config = ConfigReader.merge(arguments)[0]

            assert config['models'][0]['datasets'][0] == expected

    def test_not_overwrite_relative_paths_in_datasets_config_using_env_variable_if_commandline_provided(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'caffe'}],
            'datasets': [{
                'name': 'global_dataset',
                'dataset_meta': 'relative_annotation_path',
                'data_source': 'relative_source_path',
                'segmentation_masks_source': 'relative_source_path',
                'annotation': 'relative_annotation_path'
            }]
        }]}

        mocker.patch(self.module + '._read_configs', return_value=(
            None, local_config
        ))
        expected = copy.deepcopy(local_config['models'][0]['datasets'][0])
        with mock_filesystem(['source/']) as prefix:
            mocker.patch('os.environ.get', return_value=str(prefix))
            expected['dataset_meta'] = prefix / 'relative_annotation_path'
            expected['segmentation_masks_source'] = prefix / 'relative_source_path'
            expected['data_source'] = prefix / 'relative_source_path'
            expected['annotation'] = prefix / 'relative_annotation_path'
            expected['dataset_meta'] = prefix / 'relative_annotation_path'

            arguments = copy.deepcopy(self.arguments)
            arguments.bitstreams = None
            arguments.extensions = None
            arguments.source = None
            arguments.annotations = None

            config = ConfigReader.merge(arguments)[0]

            assert config['models'][0]['datasets'][0] == expected

    def test_not_modify_absolute_paths_in_datasets_config_using_command_line(self):
        local_config = {'models': [{
            'name': 'model',
            'datasets': [{
                'name': 'global_dataset',
                'dataset_meta': Path('/absolute_annotation_meta_path').absolute(),
                'data_source': Path('/absolute_source_path').absolute(),
                'annotation': Path('/absolute_annotation_path').absolute(),
            }]
        }]}

        expected = copy.deepcopy(local_config['models'][0]['datasets'][0])
        expected['annotation'] = Path('/absolute_annotation_path').absolute()
        expected['dataset_meta'] = Path('/absolute_annotation_meta_path').absolute()
        expected['data_source'] = Path('/absolute_source_path').absolute()

        ConfigReader._merge_paths_with_prefixes(self.arguments, local_config)

        assert local_config['models'][0]['datasets'][0] == expected

    def test_merge_launchers_with_definitions(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': 'model'}],
            'datasets': [{'name': 'global_dataset'}]
        }]}
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, local_config
        ))
        expected = copy.deepcopy(self.get_global_launcher('dlsdk'))
        expected['model'] = 'model'
        with mock_filesystem(['bitstreams/', 'extensions/']) as prefix:
            expected['bitstream'] = prefix / self.arguments.bitstreams / expected['bitstream']
            expected['cpu_extensions'] = prefix / self.arguments.extensions / expected['cpu_extensions']
            args = copy.deepcopy(self.arguments)
            args.model_optimizer = None
            args.converted_models = None
            args.models = None
            args.bitstreams = prefix / self.arguments.bitstreams
            args.extensions = prefix / self.arguments.extensions

            config = ConfigReader.merge(args)[0]

            assert config['models'][0]['launchers'][0] == expected

    def test_merge_launchers_with_model_is_not_modified(self, mocker):
        local_config = {'models': [{
            'name': 'model',
            'launchers': [{'framework': 'dlsdk', 'model': Path('/custom').absolute()}],
            'datasets': [{'name': 'global_dataset'}]
        }]}
        expected = copy.deepcopy(self.get_global_launcher('dlsdk'))
        expected['model'] = Path('/custom').absolute()
        mocker.patch(self.module + '._read_configs', return_value=(
            self.global_config, local_config
        ))
        with mock_filesystem(['bitstreams/', 'extensions/']) as prefix:
            expected['bitstream'] = prefix / self.arguments.bitstreams / expected['bitstream']
            expected['cpu_extensions'] = prefix / self.arguments.extensions / expected['cpu_extensions']
            args = copy.deepcopy(self.arguments)
            args.model_optimizer = None
            args.converted_models = None
            args.models = None
            args.bitstreams = prefix / self.arguments.bitstreams
            args.extensions = prefix / self.arguments.extensions

            config = ConfigReader.merge(args)[0]

        assert config['models'][0]['launchers'][0] == expected

    def test_expand_relative_paths_in_launchers_config_using_command_line(self, mocker):
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
            }],
            'datasets': [{'name': 'dataset'}]
        }]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        with mock_filesystem(['bitstreams/', 'extensions/', 'models/']) as prefix:
            expected = copy.deepcopy(local_config['models'][0]['launchers'][0])
            expected['model'] = prefix / self.arguments.models / 'relative_model_path'
            expected['caffe_model'] = prefix / self.arguments.models / 'relative_model_path'
            expected['tf_model'] = prefix / self.arguments.models / 'relative_model_path'
            expected['weights'] = prefix / self.arguments.models / 'relative_weights_path'
            expected['caffe_weights'] = prefix / self.arguments.models / 'relative_weights_path'
            expected['mxnet_weights'] = prefix / self.arguments.models / 'relative_weights_path'
            expected['cpu_extensions'] = prefix / self.arguments.extensions / 'relative_extensions_path'
            expected['gpu_extensions'] = prefix / self.arguments.extensions / 'relative_extensions_path'
            expected['bitstream'] = prefix / self.arguments.bitstreams / 'relative_bitstreams_path'
            args = copy.deepcopy(self.arguments)
            args.model_optimizer = None
            args.converted_models = None
            args.bitstreams = prefix / self.arguments.bitstreams
            args.extensions = prefix / self.arguments.extensions
            args.models = prefix / self.arguments.models

            config = ConfigReader.merge(args)[0]

            assert config['models'][0]['launchers'][0] == expected

    def test_both_launchers_are_filtered_by_target_tags_if_tags_not_provided_in_config(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
            },
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU',
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        arguments = copy.deepcopy(self.arguments)
        arguments.target_tags = ['some_tag']
        arguments.extensions = None
        arguments.bitstreams = None

        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))

        with pytest.warns(Warning):
            config = ConfigReader.merge(arguments)[0]
        assert len(config['models']) == 0

    def test_launcher_is_not_filtered_by_the_same_tag(self, mocker):
        config_launchers = [{
            'framework': 'dlsdk',
            'tags': ['some_tag'],
            'model': Path('/absolute_path1').absolute(),
            'weights': Path('/absolute_path1').absolute(),
            'adapter': 'classification',
            'device': 'CPU',
            '_model_optimizer': self.arguments.model_optimizer,
        }]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_tags = ['some_tag']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers[0] == config_launchers[0]

    def test_both_launchers_are_not_filtered_by_the_same_tag(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['some_tag'],
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'dlsdk',
                'tags': ['some_tag'],
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_tags = ['some_tag']

        config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 2
        assert len(config['models'][0]['launchers']) == 1
        assert len(config['models'][1]['launchers']) == 1
        launchers = [config['models'][0]['launchers'][0], config['models'][1]['launchers'][0]]
        assert launchers == config_launchers

    def test_both_launchers_are_filtered_by_another_tag(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['some_tag'],
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'dlsdk',
                'tags': ['some_tag'],
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.target_tags = ['other_tag']

        with pytest.warns(Warning):
            config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 0

    def test_only_appropriate_launcher_is_filtered_by_another_tag(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['tag1'],
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'caffe',
                'tags': ['tag2'],
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_tags = ['tag2']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[1]

    def test_only_appropriate_launcher_is_filtered_by_another_tag_if_provided_several_target_tags(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['tag1'],
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'caffe',
                'tags': ['tag2'],
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_tags = ['tag2', 'tag3']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[1]

    def test_launcher_with_several_tags_contained_at_least_one_from_target_tegs_is_not_filtered(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['tag1', 'tag2'],
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_tags = ['tag2']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[0]

    def test_both_launchers_with_different_tags_are_not_filtered_by_the_same_tags(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'tags': ['tag1'],
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'dlsdk',
                'tags': ['tag2'],
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_tags = ['tag1', 'tag2']

        config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 2
        assert len(config['models'][0]['launchers']) == 1
        assert len(config['models'][1]['launchers']) == 1
        launchers = [config['models'][0]['launchers'][0], config['models'][1]['launchers'][0]]
        assert launchers == config_launchers

    def test_launcher_is_not_filtered_by_the_same_framework(self, mocker):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': Path('/absolute_path1').absolute(),
            'weights': Path('/absolute_path1').absolute(),
            'adapter': 'classification',
            'device': 'CPU',
            '_model_optimizer': self.arguments.model_optimizer,
        }]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_framework = 'dlsdk'

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_both_launchers_are_not_filtered_by_the_same_framework(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_framework = 'dlsdk'

        config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 2
        assert len(config['models'][0]['launchers']) == 1
        assert len(config['models'][1]['launchers']) == 1
        launchers = [config['models'][0]['launchers'][0], config['models'][1]['launchers'][0]]
        assert launchers == config_launchers

    def test_launcher_is_filtered_by_another_framework(self, mocker):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': Path('/absolute_path').absolute(),
            'weights': Path('/absolute_path').absolute(),
            'adapter': 'classification',
            '_model_optimizer': self.arguments.model_optimizer,
        }]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_framework = 'caffe'

        with pytest.warns(Warning):
            config = ConfigReader.merge(args)[0]
        assert len(config['models']) == 0

    def test_both_launchers_are_filtered_by_another_framework(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU',
                '_model_optimizer': self.arguments.model_optimizer,
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_framework = 'caffe'

        with pytest.warns(Warning):
            config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 0

    def test_only_appropriate_launcher_is_filtered_by_another_framework(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_framework = 'caffe'

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[1]

    def test_launcher_is_not_filtered_by_the_same_device(self, mocker):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': Path('/absolute_path1').absolute(),
            'weights': Path('/absolute_path1').absolute(),
            'adapter': 'classification',
            'device': 'CPU',
            '_model_optimizer': self.arguments.model_optimizer,
        }]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_devices = ['CPU']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert launchers == config_launchers

    def test_both_launchers_are_not_filtered_by_the_same_device(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_devices = ['CPU']

        config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 2
        assert len(config['models'][0]['launchers']) == 1
        assert len(config['models'][1]['launchers']) == 1
        launchers = [config['models'][0]['launchers'][0], config['models'][1]['launchers'][0]]
        assert launchers == config_launchers

    def test_launcher_is_filtered_by_another_device(self, mocker):
        config_launchers = [{
            'framework': 'dlsdk',
            'model': Path('/absolute_path1').absolute(),
            'weights': Path('/absolute_path1').absolute(),
            'adapter': 'classification',
            'device': 'CPU',
            '_model_optimizer': self.arguments.model_optimizer,
        }]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.target_devices = ['GPU']

        with pytest.warns(Warning):
            config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 0

    def test_both_launchers_are_filtered_by_another_device(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.model_optimizer = None
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_devices = ['GPU']

        with pytest.warns(Warning):
            config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 0

    def test_only_appropriate_launcher_is_filtered_by_another_device(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.target_devices = ['GPU']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[1]

    def test_only_appropriate_launcher_is_filtered_by_user_input_devices(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'HETERO:CPU,GPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU',
            }
        ]

        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_devices = ['GPU', 'CPU']

        config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 2
        assert len(config['models'][0]['launchers']) == 1
        assert len(config['models'][1]['launchers']) == 1
        launchers = [config['models'][0]['launchers'][0], config['models'][1]['launchers'][0]]
        assert launchers == [config_launchers[0], config_launchers[2]]

    def test_both_launchers_are_filtered_by_other_devices(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.target_devices = ['FPGA', 'MYRIAD']
        args.extensions = None
        args.bitstreams = None

        with pytest.warns(Warning):
            config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 0

    def test_both_launchers_are_not_filtered_by_same_devices(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_devices = ['GPU', 'CPU']

        config = ConfigReader.merge(args)[0]

        assert len(config['models']) == 2
        assert len(config['models'][0]['launchers']) == 1
        assert len(config['models'][1]['launchers']) == 1
        launchers = [config['models'][0]['launchers'][0], config['models'][1]['launchers'][0]]
        assert launchers == config_launchers

    def test_launcher_is_not_filtered_by_device_with_tail(self, mocker):
        config_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'GPU'
            }
        ]
        local_config = {'models': [{'name': 'name', 'launchers': config_launchers, 'datasets': [{'name': 'dataset'}]}]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.extensions = None
        args.bitstreams = None
        args.target_devices = ['CPU', 'GPU_unexpected_tail']

        config = ConfigReader.merge(args)[0]

        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert launchers[0] == config_launchers[0]

    def test_all_model_launchers_filtered_in_config_with_several_models(self, mocker):
        model1_launchers = [
            {
                'framework': 'dlsdk',
                'model': Path('/absolute_path1').absolute(),
                'weights': Path('/absolute_path1').absolute(),
                'adapter': 'classification',
                'device': 'CPU',
                '_model_optimizer': self.arguments.model_optimizer,
            },
            {
                'framework': 'caffe',
                'model': Path('/absolute_path2').absolute(),
                'weights': Path('/absolute_path2').absolute(),
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        model2_launchers = [
            {
                'framework': 'tf',
                'model': Path('/absolute_path3').absolute(),
                'adapter': 'classification',
                'device': 'CPU'
            }
        ]
        local_config = {'models': [
            {'name': 'model_1', 'launchers': model1_launchers, 'datasets': [{'name': 'dataset'}]},
            {'name': 'model_2', 'launchers': model2_launchers, 'datasets': [{'name': 'dataset'}]}
        ]}
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.converted_models = None
        args.target_framework = 'tf'
        with pytest.warns(Warning):
            config = ConfigReader.merge(args)[0]
        assert len(config['models']) == 1
        assert config['models'][0]['name'] == 'model_2'
        assert config['models'][0]['launchers'] == model2_launchers

    def test_replace_empty_device_by_target_in_models_mode(self, mocker):
        local_config = {
            'models': [{
                'name': 'model',
                'launchers': [{
                    'framework': 'caffe',
                    'model': Path('/absolute_path2').absolute(),
                    'weights': Path('/absolute_path2').absolute(),
                    'adapter': 'classification',
            }],
                'datasets': [{'name': 'dataset'}]}]
        }
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.target_devices = ['CPU']
        config, _ = ConfigReader.merge(args)
        launchers = config['models'][0]['launchers']
        assert len(launchers) == 1
        assert 'device' in launchers[0]
        assert launchers[0]['device'].upper() == 'CPU'

    def test_replace_empty_device_by_several_targets_in_models_mode(self, mocker):
        local_config = {
            'models': [{
                'name': 'model',
                'launchers': [{
                    'framework': 'caffe',
                    'model': Path('/absolute_path2').absolute(),
                    'weights': Path('/absolute_path2').absolute(),
                    'adapter': 'classification',
            }],
                'datasets': [{'name': 'dataset'}]}]
        }
        mocker.patch(self.module + '._read_configs', return_value=(None, local_config))
        args = copy.deepcopy(self.arguments)
        args.target_devices = ['CPU', 'GPU']
        config, _ = ConfigReader.merge(args)
        assert len(config['models']) == 2
        assert len(config['models'][0]['launchers']) == 1
        assert len(config['models'][1]['launchers']) == 1
        launchers = [config['models'][0]['launchers'][0], config['models'][1]['launchers'][0]]
        assert 'device' in launchers[0]
        assert 'device' in launchers[1]
        assert launchers[0]['device'].upper() == 'CPU'
        assert launchers[1]['device'].upper() == 'GPU'

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
