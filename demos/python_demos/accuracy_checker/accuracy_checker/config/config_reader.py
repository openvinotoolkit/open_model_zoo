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

import warnings

from ..utils import read_yaml, to_lower_register


class ConfigReader:
    """
    Class for parsing input config
    """
    @staticmethod
    def merge(arguments):
        """
        Args:
            arguments: command-line arguments
        Returns:
            dictionary containing configuration
        """
        global_config, local_config = ConfigReader._read_configs(arguments)
        if not local_config:
            raise ValueError('Missing local config')

        ConfigReader._check_local_config(local_config)
        ConfigReader._prepare_global_configs(global_config)

        config = ConfigReader._merge_configs(global_config, local_config)

        ConfigReader._provide_cmd_arguments(arguments, config)
        ConfigReader._merge_paths_with_prefixes(arguments, config)
        ConfigReader._filter_launchers(config, arguments)

        return config

    @staticmethod
    def _read_configs(arguments):
        global_config = read_yaml(arguments.definitions) if arguments.definitions else None
        local_config = read_yaml(arguments.config)

        return global_config, local_config

    @staticmethod
    def _check_local_config(config):
        models = config.get('models')
        if not models:
            raise ValueError('Missed "{}" in local config'.format('models'))

        required_entries = ['name', 'launchers', 'datasets']
        for model in models:
            for entry in required_entries:
                config_entry = model.get(entry)
                if config_entry:
                    continue

                raise ValueError('Each model must specify {}'.format(required_entries))

            for dataset in model['datasets']:
                required = ['name']

                missed = [entry for entry in required if entry not in dataset]
                if not missed:
                    continue

                message = 'Model "{}" must specify "{}" for each {}'.format(model['name'], required, 'dataset')
                raise ValueError(message)

    @staticmethod
    def _prepare_global_configs(global_configs):
        if not global_configs or 'datasets' not in global_configs:
            return
        datasets = global_configs['datasets']

        def merge(local_entries, global_entries, identifier):
            if not local_entries or not global_entries:
                return

            for i, local in enumerate(local_entries):
                local_identifier = local.get(identifier)
                if not local_identifier:
                    continue

                local_entries[i] = ConfigReader._merge_configs_by_identifier(global_entries, local, identifier)

        for dataset in datasets:
            merge(dataset.get('preprocessing'), global_configs.get('preprocessing'), 'type')
            merge(dataset.get('metrics'), global_configs.get('metrics'), 'type')
            merge(dataset.get('postprocessing'), global_configs.get('postprocessing'), 'type')

    @staticmethod
    def _merge_configs(global_configs, local_config):
        config = copy.deepcopy(local_config)
        if not global_configs:
            return config

        models = config.get('models')
        for model in models:
            for i, launcher_entry in enumerate(model['launchers']):
                model['launchers'][i] = ConfigReader._merge_configs_by_identifier(
                    global_configs['launchers'], launcher_entry, 'framework'
                )

            for i, dataset in enumerate(model['datasets']):
                model['datasets'][i] = ConfigReader._merge_configs_by_identifier(
                    global_configs['datasets'], dataset, 'name'
                )

        return config

    @staticmethod
    def _merge_configs_by_identifier(global_config, local_config, identifier):
        local_identifier = local_config.get(identifier)
        if not local_identifier:
            return local_config

        matched = []
        for config in global_config:
            global_identifier = config.get(identifier)
            if not global_identifier:
                continue

            if global_identifier != local_identifier:
                continue

            matched.append(config)

        fallback = matched[0] if matched else {}

        config = copy.deepcopy(fallback)
        for key, value in local_config.items():
            config[key] = value

        return config

    @staticmethod
    def _merge_paths_with_prefixes(arguments, config):
        args = arguments if isinstance(arguments, dict) else vars(arguments)
        entries_paths = {
            'launchers': {
                'model': 'models',
                'weights': 'models',
                'caffe_model': 'models',
                'caffe_weights': 'models',
                'tf_model': 'models',
                'mxnet_weights': 'models',
                'onnx_model': 'models',
                'kaldi_model': 'models',
                'cpu_extensions': 'extensions',
                'gpu_extensions': 'extensions',
                'bitstream': 'bitstreams'
            },
            'datasets': {
                'data_source': 'source',
                'segmentation_masks_source': 'source',
                'annotation': 'annotations',
                'dataset_meta': 'annotations'
            }
        }

        def merge_entry_paths(keys, value):
            for field, argument in keys.items():
                if field not in value:
                    continue

                config_path = Path(value[field])
                if config_path.is_absolute():
                    continue

                value[field] = args[argument] / config_path

        for model in config['models']:
            for entry, command_line_arg in entries_paths.items():
                if entry not in model:
                    continue

                for config_entry in model[entry]:
                    merge_entry_paths(command_line_arg, config_entry)

    @staticmethod
    def _provide_cmd_arguments(arguments, config):
        for model in config['models']:
            for launcher_entry in model['launchers']:
                if launcher_entry['framework'].lower() != 'dlsdk':
                    continue

                converted_models = arguments.converted_models if arguments.converted_models else arguments.models

                launcher_entry['_converted_models'] = converted_models
                launcher_entry['_models_prefix'] = arguments.models
                launcher_entry['_model_optimizer'] = arguments.model_optimizer
                launcher_entry['_tf_custom_config_dir'] = arguments.tf_custom_op_config

    @staticmethod
    def _filter_launchers(config, arguments):
        def filtered(launcher, target_devices):
            config_framework = launcher['framework'].lower()
            target_framework = (args.get('target_framework') or config_framework).lower()
            if config_framework != target_framework:
                return True

            return target_devices and launcher.get('device', '').lower() not in target_devices

        args = arguments if isinstance(arguments, dict) else vars(arguments)
        target_devices = to_lower_register(args.get('target_devices') or [])
        for model in config['models']:
            launchers = model['launchers']
            launchers = [launcher for launcher in launchers if not filtered(launcher, target_devices)]

            if not launchers:
                warnings.warn('Model "{}" has no launchers'.format(model['name']))

            model['launchers'] = launchers
