"""
Copyright (c) 2019 Intel Corporation

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
import os

import warnings

from ..utils import read_yaml, to_lower_register, contains_any
from .config_validator import ConfigError

ENTRIES_PATHS = {
    'launchers': {
        'cpu_extensions': 'extensions',
        'gpu_extensions': 'extensions',
        'bitstream': 'bitstreams',
        'affinity_map': 'affinity_map',
        'predictions': 'source'
    },
    'datasets': {
        'segmentation_masks_source': 'source',
        'annotation': 'annotations',
        'dataset_meta': 'annotations',
        'data_source': 'source',
    },
}

PREPROCESSING_PATHS = {
    'mask_dir': 'source',
    'vocabulary_file': ['model_attributes', 'models']
}

ADAPTERS_PATHS = {
    'vocabulary_file': ['model_attributes', 'models']
}

ANNOTATION_CONVERSION_PATHS = {
    'vocab_file': ['model_attributes', 'source', 'models'],
    'merges_file': ['model_attributes', 'source', 'models']
}

LIST_ENTRIES_PATHS = {
        'model': 'models',
        'weights': 'models',
        'color_coeff': ['model_attributes', 'models'],
        'caffe_model': 'models',
        'caffe_weights': 'models',
        'tf_model': 'models',
        'tf_meta': 'models',
        'mxnet_weights': 'models',
        'onnx_model': 'models',
        'kaldi_model': 'models',
}

COMMAND_LINE_ARGS_AS_ENV_VARS = {
    'source': 'DATA_DIR',
    'annotations': 'ANNOTATIONS_DIR',
    'bitstreams': 'BITSTREAMS_DIR',
    'models': 'MODELS_DIR',
    'extensions': 'EXTENSIONS_DIR',
    'model_attributes': 'MODEL_ATTRIBUTES_DIR',
}
DEFINITION_ENV_VAR = 'DEFINITIONS_FILE'
CONFIG_SHARED_PARAMETERS = ['bitstream']
ACCEPTABLE_MODEL = [
    'caffe_model', 'caffe_weights',
    'tf_model', 'tf_meta',
    'mxnet_weights',
    'onnx_model',
    'kaldi_model',
    'model'
]


class ConfigReader:
    """
    Class for parsing input config.
    """

    @staticmethod
    def merge(arguments):
        """
        Args:
            arguments: command-line arguments.
        Returns:
            dictionary containing configuration.
        """

        global_config, local_config = ConfigReader._read_configs(arguments)
        if not local_config:
            raise ConfigError('Missing local config')

        mode = ConfigReader.check_local_config(local_config)
        ConfigReader._prepare_global_configs(global_config)

        config = ConfigReader._merge_configs(global_config, local_config, arguments, mode)
        ConfigReader.process_config(config, mode, arguments)

        return config, mode

    @staticmethod
    def process_config(config, mode='models', arguments=None):
        if arguments is None:
            arguments = dict()
        ConfigReader._merge_paths_with_prefixes(arguments, config, mode)
        ConfigReader._provide_cmd_arguments(arguments, config, mode)
        ConfigReader._filter_launchers(config, arguments, mode)
        ConfigReader._separate_evaluations(config, mode)
        ConfigReader._previous_configuration_parameters_sharing(config, mode)

    @staticmethod
    def _read_configs(arguments):
        local_config = read_yaml(arguments.config)
        definitions = os.environ.get(DEFINITION_ENV_VAR) or local_config.get('global_definitions')
        if definitions:
            definitions = read_yaml(Path(arguments.config).parent / definitions)
        global_config = read_yaml(arguments.definitions) if arguments.definitions else definitions

        return global_config, local_config

    @staticmethod
    def check_local_config(config):
        def _is_requirements_missed(target, requirements):
            return list(filter(lambda entry: not target.get(entry), requirements))

        def _check_models_config(config):
            models = config.get('models')
            if not models:
                raise ConfigError('Missed "{}" in local config'.format('models'))

            required_model_entries = ['name', 'launchers', 'datasets']
            required_dataset_entries = ['name']
            required_dataset_error = 'Model {} must specify {} for each dataset'
            for model in models:
                if _is_requirements_missed(model, required_model_entries):
                    raise ConfigError('Each model must specify {}'.format(', '.join(required_model_entries)))
                datasets = model['datasets'].values() if isinstance(model['datasets'], dict) else model['datasets']
                if list(filter(lambda entry: _is_requirements_missed(entry, required_dataset_entries), datasets)):
                    raise ConfigError(required_dataset_error.format(model['name'], ', '.join(required_dataset_entries)))

        def _check_module_config(config):
            required_entries = ['name', 'module']
            evaluations = config['evaluations']
            if not evaluations:
                raise ConfigError('Missed "{}" in local config'.format('evaluations'))
            for evaluation in evaluations:
                if _is_requirements_missed(evaluation, required_entries):
                    raise ConfigError('Each evaluations must specify {}'.format(', '.join(required_entries)))

        config_checkers = {
            'evaluations': _check_module_config,
            'models': _check_models_config,
        }

        if not isinstance(config, dict):
            raise ConfigError('local config should has dictionary based structure')

        eval_mode = get_mode(config)
        config_checker_func = config_checkers.get(eval_mode)
        if config_checker_func is None:
            raise ConfigError(
                'Accuracy Checker {} mode is not supported. Please select between evaluations and models.'. format(
                    eval_mode))
        config_checker_func(config)

        return eval_mode

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
    def _merge_models_config(global_configs, local_config, arguments):
        config = copy.deepcopy(local_config)
        if not global_configs:
            return config

        models = config['models']
        for model in models:
            if 'launchers' in global_configs:
                for i, launcher_entry in enumerate(model['launchers']):
                    model['launchers'][i] = ConfigReader._merge_configs_by_identifier(
                        global_configs['launchers'], launcher_entry, 'framework'
                    )
            if 'datasets' in global_configs:
                datasets_iterator = (
                    model['datasets'].items() if isinstance(model['datasets'], dict)
                    else enumerate(model['datasets'])
                )
                for i, dataset in datasets_iterator:
                    model['datasets'][i] = ConfigReader._merge_configs_by_identifier(
                        global_configs['datasets'], dataset, 'name'
                    )

        config['models'] = models
        return config

    @staticmethod
    def _merge_module_config(global_config, local_config, args):

        config = copy.deepcopy(local_config)
        if not global_config:
            return config

        for evaluation in config['evaluations']:
            if 'module_config' not in evaluation:
                continue
            module_config = evaluation['module_config']
            if 'launchers' in module_config and 'launchers' in global_config:
                for i, launcher_entry in enumerate(module_config['launchers']):
                    module_config['launchers'][i] = ConfigReader._merge_configs_by_identifier(
                        global_config['launchers'], launcher_entry, 'framework'
                    )
            if 'datasets' in module_config and 'datasets' in global_config:
                datasets_iterator = (
                    module_config['datasets'].items() if isinstance(module_config['datasets'], dict)
                    else enumerate(module_config['datasets'])
                )
                for i, dataset in datasets_iterator:
                    module_config['datasets'][i] = ConfigReader._merge_configs_by_identifier(
                        global_config['datasets'], dataset, 'name'
                    )

        return config

    @staticmethod
    def _merge_configs(global_configs, local_config, arguments, mode='models'):
        functors_by_mode = {
            'models': ConfigReader._merge_models_config,
            'evaluations': ConfigReader._merge_module_config
        }

        return functors_by_mode[mode](global_configs, local_config, arguments)

    @staticmethod
    def _merge_configs_by_identifier(global_config, local_config, identifier):
        local_identifier = local_config.get(identifier)
        if local_identifier is None:
            return local_config

        matched = []
        for config in global_config:
            global_identifier = config.get(identifier)
            if global_identifier is None:
                continue

            if global_identifier != local_identifier:
                continue

            matched.append(config)

        config = copy.deepcopy(matched[0] if matched else {})
        for key, value in local_config.items():
            config[key] = value

        return config

    @staticmethod
    def _merge_paths_with_prefixes(arguments, config, mode='models'):
        args = arguments if isinstance(arguments, dict) else vars(arguments)
        for argument, env_var in COMMAND_LINE_ARGS_AS_ENV_VARS.items():
            if argument not in args or args[argument] is None:
                env_var_value = os.environ.get(env_var)
                if env_var_value is not None:
                    args[argument] = Path(env_var_value)

        def process_models(config, entries_paths):
            for model in config['models']:
                process_config(model, entries_paths, args)

        def process_modules(config, entries_paths):
            for evaluation in config['evaluations']:
                module_config = evaluation.get('module_config')
                if not module_config:
                    continue
                process_config(module_config, entries_paths, args)
                if 'network_info' in module_config:
                    networks_info = module_config['network_info']
                    if isinstance(networks_info, dict):
                        for _, params in networks_info.items():
                            entries_paths['launchers'].update(LIST_ENTRIES_PATHS)
                            merge_entry_paths(entries_paths['launchers'], params, args)
                    if isinstance(networks_info, list):
                        merge_entry_paths(entries_paths['launchers'], networks_info, args)

        functors_by_mode = {
            'models': process_models,
            'evaluations': process_modules
        }

        processing_func = functors_by_mode[mode]
        processing_func(config, ENTRIES_PATHS.copy())

    @staticmethod
    def _provide_cmd_arguments(arguments, config, mode):
        def _add_subset_specific_arg(dataset_entry):
            if 'shuffle' in arguments and arguments.shuffle is not None:
                dataset_entry['shuffle'] = arguments.shuffle

            if 'subsample_size' in arguments and arguments.subsample_size is not None:
                dataset_entry['subsample_size'] = arguments.subsample_size

        def merge_models(config, arguments, update_launcher_entry):
            def provide_models(launchers):
                if 'models' not in arguments or not arguments.models:
                    return launchers
                model_paths = arguments.models
                updated_launchers = []
                model_paths = [model_paths] if not isinstance(model_paths, list) else model_paths
                for launcher in launchers:
                    if contains_any(launcher, ACCEPTABLE_MODEL):
                        updated_launchers.append(launcher)
                        continue
                    for model_path in model_paths:
                        copy_launcher = copy.deepcopy(launcher)
                        copy_launcher['model'] = model_path
                        if launcher['framework'] == 'dlsdk' and 'model_is_blob' in arguments:
                            copy_launcher['_model_is_blob'] = arguments.model_is_blob
                        updated_launchers.append(copy_launcher)
                return updated_launchers

            for model in config['models']:
                for launcher_entry in model['launchers']:
                    merge_dlsdk_launcher_args(arguments, launcher_entry, update_launcher_entry)
                model['launchers'] = provide_models(model['launchers'])
                for dataset_entry in model['datasets']:
                    _add_subset_specific_arg(dataset_entry)
                    if 'ie_preprocessing' in arguments and arguments.ie_preprocessing:
                        dataset_entry['_ie_preprocessing'] = arguments.ie_preprocessing

        def merge_modules(config, arguments, update_launcher_entry):
            for evaluation in config['evaluations']:
                module_config = evaluation.get('module_config')
                if not module_config:
                    continue
                if 'models' in arguments and arguments.models:
                    module_config['_models'] = arguments.models
                    if 'model_is_blob' in arguments:
                        module_config['_model_is_blob'] = arguments.model_is_blob
                if 'launchers' not in module_config:
                    continue
                for launcher in module_config['launchers']:
                    merge_dlsdk_launcher_args(arguments, launcher, update_launcher_entry)
                for dataset in module_config['datasets']:
                    _add_subset_specific_arg(dataset)

        functors_by_mode = {
            'models': merge_models,
            'evaluations': merge_modules
        }

        additional_keys = [
            'model_optimizer', 'tf_custom_op_config_dir',
            'tf_obj_detection_api_pipeline_config_path',
            'transformations_config_dir',
            'cpu_extensions_mode', 'vpu_log_level', 'device_config'
        ]
        arguments_dict = arguments if isinstance(arguments, dict) else vars(arguments)
        update_launcher_entry = {}

        for key in additional_keys:
            value = arguments_dict.get(key)
            if value:
                update_launcher_entry['_{}'.format(key)] = value

        return functors_by_mode[mode](config, arguments, update_launcher_entry)

    @staticmethod
    def _filter_launchers(config, arguments, mode='models'):
        functors_by_mode = {
            'models': filter_models,
            'evaluations': filter_modules
        }

        args = arguments if isinstance(arguments, dict) else vars(arguments)
        target_devices = to_lower_register(args.get('target_devices') or [])
        filtering_mode = functors_by_mode[mode]
        filtering_mode(config, target_devices, args)

    @staticmethod
    def _separate_evaluations(config, mode='models'):
        def _separate_models_evaluations(models_config):
            evaluations = []
            for model in models_config['models']:
                launchers = model['launchers']
                datasets = model['datasets']
                if not launchers:
                    continue
                if len(launchers) == 1 and len(datasets) == 1:
                    evaluations.append(model)
                    continue
                for launcher in model['launchers']:
                    model_evaluations = []
                    model_config_copy_launcher = copy.deepcopy(model)
                    model_config_copy_launcher['launchers'] = [launcher]

                    for dataset in model_config_copy_launcher['datasets']:
                        model_config_copy_dataset = copy.deepcopy(model_config_copy_launcher)
                        model_config_copy_dataset['datasets'] = [dataset]
                        model_evaluations.append(model_config_copy_dataset)

                    evaluations.extend(model_evaluations)

            models_config['models'] = evaluations

        def _separate_modules_evaluations(modules_config):
            evals = modules_config['evaluations']
            eval_list = []
            for evaluation in evals:
                if 'module_config' not in evaluation:
                    eval_list.append(evaluation)
                    continue
                module_config = evaluation['module_config']
                launchers = module_config.get('launchers', [])
                datasets = module_config.get('datasets', [])
                eval_config_list = []
                for launcher in launchers:
                    copy_module_config = copy.deepcopy(module_config)
                    copy_module_config['launchers'] = [launcher]
                    if not datasets:
                        eval_config_list.append(copy_module_config)
                        continue
                    for dataset in datasets:
                        copy_evaluation_for_dataset = copy.deepcopy(copy_module_config)
                        copy_evaluation_for_dataset['datasets'] = [dataset]
                        eval_config_list.append(copy_evaluation_for_dataset)
                for eval_config in eval_config_list:
                    copy_evaluation = copy.deepcopy(evaluation)
                    copy_evaluation['module_config'] = eval_config
                    eval_list.append(copy_evaluation)

            modules_config['evaluations'] = eval_list

        mode_func = {
            'models': _separate_models_evaluations,
            'evaluations': _separate_modules_evaluations
        }

        separator = mode_func.get(mode)
        if not separator:
            return
        separator(config)

    @staticmethod
    def _previous_configuration_parameters_sharing(config, mode='models'):
        def _share_params_models(models_config):
            shared_params = {parameter: None for parameter in CONFIG_SHARED_PARAMETERS}
            for model in models_config['models']:
                launchers = model['launchers']
                if not launchers:
                    continue
                for launcher in model['launchers']:
                    for parameter in CONFIG_SHARED_PARAMETERS:
                        if parameter in launcher:
                            if shared_params[parameter] is not None:
                                launcher['_prev_{}'.format(parameter)] = shared_params[parameter]
                            shared_params[parameter] = launcher[parameter]

        def _share_params_modules(modules_config):
            shared_params = {parameter: None for parameter in CONFIG_SHARED_PARAMETERS}
            for evaluation in modules_config['evaluations']:
                if 'module_config' not in evaluation:
                    continue
                launchers = evaluation['module_config'].get('launchers')
                for launcher in launchers:
                    for parameter in CONFIG_SHARED_PARAMETERS:
                        if parameter in launcher:
                            if shared_params[parameter] is not None:
                                launcher['_prev_{}'.format(parameter)] = shared_params[parameter]
                            shared_params[parameter] = launcher[parameter]

        mode_func = {
            'models': _share_params_models,
            'evaluations': _share_params_modules,
        }

        processor = mode_func.get(mode)
        if not processor:
            return
        processor(config)

    @staticmethod
    def convert_paths(config):
        mode = 'evaluations' if 'evaluations' in config else 'models'
        definitions = os.environ.get(DEFINITION_ENV_VAR)
        if definitions:
            definitions = read_yaml(Path(definitions))
            ConfigReader._prepare_global_configs(definitions)
            config = ConfigReader._merge_configs(definitions, config, {}, mode)
        if COMMAND_LINE_ARGS_AS_ENV_VARS['source'] in os.environ:
            ConfigReader._merge_paths_with_prefixes({}, config, mode)

        def convert_launcher_paths(launcher_config):
            for key, path in launcher_config.items():
                if key not in ENTRIES_PATHS['launchers']:
                    continue
                launcher_config[key] = Path(path)
            adapter_config = launcher_config.get('adapter')
            if isinstance(adapter_config, dict):
                command_line_adapter = (create_command_line_mapping(adapter_config, None))
                for arg in command_line_adapter:
                    adapter_config[arg] = Path(adapter_config[arg])

        def convert_dataset_paths(dataset_config):
            conversion_config = dataset_config.get('annotation_conversion')
            if conversion_config:
                command_line_conversion = (create_command_line_mapping(conversion_config, None))
                for conversion_path in command_line_conversion:
                    conversion_config[conversion_path] = Path(conversion_config[conversion_path])

            if 'preprocessing' in dataset_config:
                for preprocessor in dataset_config['preprocessing']:
                    path_preprocessing = (create_command_line_mapping(preprocessor, None))
                    for path in path_preprocessing:
                        preprocessor[path] = Path(preprocessor[path])

            for key, path in dataset_config.items():
                if key not in ENTRIES_PATHS['datasets']:
                    continue
                dataset_config[key] = Path(path)

        if mode == 'models':
            for model in config['models']:
                for launcher_config in model['launchers']:
                    convert_launcher_paths(launcher_config)
                datasets = model['datasets'].values() if isinstance(model['datasets'], dict) else model['datasets']
                for dataset_config in datasets:
                    convert_dataset_paths(dataset_config)
        else:
            for evaluation in config['evaluations']:
                module_config = evaluation.get('module_config', {})
                for launcher_config in module_config.get('launchers', []):
                    convert_launcher_paths(launcher_config)
                d_config = module_config.get('datasets')
                if d_config:
                    datasets = d_config.values() if isinstance(d_config, dict) else d_config
                    for dataset_config in datasets:
                        convert_dataset_paths(dataset_config)

        return config


def create_command_line_mapping(config, default_value, value_map=None):
    mapping = {}
    value_map = value_map or {}
    for key in config:
        if key.endswith('file') or key.endswith('dir'):
            mapping[key] = value_map.get(key, default_value)

    return mapping


def filtered(launcher, targets, args):
    target_tags = args.get('target_tags') or []
    if target_tags:
        if not contains_any(target_tags, launcher.get('tags', [])):
            return True

    config_framework = launcher['framework'].lower()
    target_framework = (args.get('target_framework') or config_framework).lower()
    if config_framework != target_framework:
        return True

    return targets and launcher.get('device', '').lower() not in targets


def filter_models(config, target_devices, args):
    models_after_filtration = []
    for model in config['models']:
        launchers_after_filtration = []
        launchers = model['launchers']
        for launcher in launchers:
            if 'device' not in launcher and target_devices:
                for device in target_devices:
                    launcher_with_device = copy.deepcopy(launcher)
                    launcher_with_device['device'] = device
                    if not filtered(launcher_with_device, target_devices, args):
                        launchers_after_filtration.append(launcher_with_device)
                continue
            if not filtered(launcher, target_devices, args):
                launchers_after_filtration.append(launcher)

        if not launchers_after_filtration:
            warnings.warn('Model "{}" has no launchers'.format(model['name']))
            continue

        model['launchers'] = launchers_after_filtration
        models_after_filtration.append(model)

    config['models'] = models_after_filtration


def filter_modules(config, target_devices, args):
    filtered_evals = []
    for evaluation in config['evaluations']:
        if 'module_config' not in evaluation or 'launchers' not in evaluation['module_config']:
            if target_devices:
                warnings.warn(
                    'Information about launcher is not provided in config for {}. '
                    'Filtration can not be done'.format(evaluation['name'])
                )
            filtered_evals.append(evaluation)
            continue
        module_config = evaluation['module_config']
        launchers = module_config['launchers']
        if target_devices:
            launchers_without_device = [launcher for launcher in launchers if 'device' not in launcher]
            for launcher in launchers_without_device:
                for device in target_devices:
                    launcher_with_device = copy.deepcopy(launcher)
                    launcher_with_device['device'] = device
                    launchers.append(launcher_with_device)
        launchers = [
            launcher for launcher in launchers if not filtered(launcher, target_devices, args)
        ]
        if not launchers:
            warnings.warn('Model "{}" has no launchers'.format(evaluation['name']))
        evaluation['module_config']['launchers'] = launchers
        filtered_evals.append(evaluation)
    config['evaluations'] = filtered_evals


def process_config(
        config_item, entries_paths, args, dataset_identifier='datasets',
        launchers_identifier='launchers', identifers_mapping=None, pipeline=False
):
    def process_dataset(datasets_configs):
        for datasets_config in datasets_configs:
            annotation_conversion_config = datasets_config.get('annotation_conversion')
            if annotation_conversion_config:
                command_line_conversion = (create_command_line_mapping(annotation_conversion_config,
                                                                       'source', ANNOTATION_CONVERSION_PATHS))
                merge_entry_paths(command_line_conversion, annotation_conversion_config, args)
            if 'preprocessing' in datasets_config:
                for preprocessor in datasets_config['preprocessing']:
                    command_line_preprocessing = (
                        create_command_line_mapping(preprocessor, 'models', PREPROCESSING_PATHS)
                    )
                    merge_entry_paths(command_line_preprocessing, preprocessor, args)

    def process_launchers(launchers_configs):
        if not isinstance(launchers_configs, list):
            launchers_configs = [launchers_configs]

        updated_launchers = []
        for launcher_config in launchers_configs:
            if 'models' not in args or not args['models']:
                updated_launchers.append(launcher_config)
                continue
            models = args['models']
            if isinstance(models, list):
                for model_id, _ in enumerate(models):
                    new_launcher = copy.deepcopy(launcher_config)
                    merge_entry_paths(LIST_ENTRIES_PATHS, new_launcher, args, model_id)
                    adapter_config = new_launcher.get('adapter')
                    if isinstance(adapter_config, dict):
                        command_line_adapter = (create_command_line_mapping(adapter_config, 'models', ADAPTERS_PATHS))
                        merge_entry_paths(command_line_adapter, adapter_config, args, model_id)
                    if not updated_launchers or new_launcher != updated_launchers[-1]:
                        updated_launchers.append(new_launcher)
            else:
                merge_entry_paths(LIST_ENTRIES_PATHS, launcher_config, args)
                adapter_config = launcher_config.get('adapter')
                if isinstance(adapter_config, dict):
                    command_line_adapter = (create_command_line_mapping(adapter_config, 'models', ADAPTERS_PATHS))
                    merge_entry_paths(command_line_adapter, adapter_config, args)
                updated_launchers.append(launcher_config)

        return updated_launchers

    for entry, command_line_arg in entries_paths.items():
        entry_id = entry if not identifers_mapping else identifers_mapping[entry]
        if entry_id not in config_item:
            continue

        if entry_id == dataset_identifier:
            datasets_config = config_item[entry_id]
            dataset_processing_config = (
                list(datasets_config.values()) if isinstance(datasets_config, dict) and not pipeline
                else datasets_config
            )
            if not isinstance(dataset_processing_config, list):
                dataset_processing_config = [dataset_processing_config]
            process_dataset(dataset_processing_config)
            for config_entry in dataset_processing_config:
                merge_entry_paths(command_line_arg, config_entry, args)
            continue

        if entry_id == launchers_identifier:
            launchers_configs = config_item[entry_id]
            processed_launcher = process_launchers(launchers_configs)
            config_item[entry_id] = processed_launcher if not pipeline else processed_launcher[0]

        config_entries = config_item[entry_id]
        if not isinstance(config_entries, list):
            config_entries = [config_entries]
        for config_entry in config_entries:
            merge_entry_paths(command_line_arg, config_entry, args)


def merge_entry_paths(keys, value, args, value_id=0):
    for field, argument in keys.items():
        if field not in value:
            continue

        config_path = Path(value[field])
        if config_path.is_absolute():
            value[field] = Path(value[field])
            continue

        if isinstance(argument, list):
            argument = next(filter(args.get, argument), argument[-1])

        if argument not in args or not args[argument]:
            continue

        selected_argument = args[argument]
        if isinstance(selected_argument, list):
            if len(selected_argument) > 1:
                if len(selected_argument) <= value_id:
                    raise ValueError('list of arguments for {} less than number of evaluations')
                selected_argument = selected_argument[value_id]
            else:
                selected_argument = selected_argument[0]

        if not selected_argument.is_dir():
            raise ConfigError('argument: {} should be a directory'.format(argument))
        value[field] = selected_argument / config_path


def get_mode(config):
    evaluation_keys = [key for key in config if key != 'global_definitions']
    if not evaluation_keys:
        raise ConfigError('Invalid config structure. No evaluations detected.')
    if len(evaluation_keys) > 1:
        raise ConfigError('Multiple evaluation types in the one config is not supported. '
                          'Please separate on several configs.')
    return next(iter(evaluation_keys))


def merge_converted_model_path(converted_models_dir, mo_output_dir):
    if mo_output_dir:
        mo_output_dir = Path(mo_output_dir)
        if mo_output_dir.is_absolute():
            return mo_output_dir
        return converted_models_dir / mo_output_dir
    return converted_models_dir


def merge_dlsdk_launcher_args(arguments, launcher_entry, update_launcher_entry):
    def _convert_models_args(launcher_entry):
        if 'deprecated_ir_v7' in arguments and arguments.deprecated_ir_v7:
            mo_flags = launcher_entry.get('mo_flags', [])
            mo_flags.append('generate_deprecated_IR_V7')
            launcher_entry['mo_flags'] = mo_flags
        if 'converted_models' in arguments and arguments.converted_models:
            mo_params = launcher_entry.get('mo_params', {})
            mo_params.update({
                'output_dir': merge_converted_model_path(arguments.converted_models,
                                                         mo_params.get('output_dir'))
            })

            launcher_entry['mo_params'] = mo_params

        return launcher_entry

    def _fpga_specific_args(launcher_entry):
        if 'aocl' in arguments and arguments.aocl:
            launcher_entry['_aocl'] = arguments.aocl

        if 'bitstream' not in launcher_entry and 'bitstreams' in arguments and arguments.bitstreams:
            if not arguments.bitstreams.is_dir():
                launcher_entry['bitstream'] = arguments.bitstreams

    def _async_evaluation_args(launcher_entry):
        if 'async_mode' in arguments:
            launcher_entry['async_mode'] = arguments.async_mode

        if 'num_requests' in arguments and arguments.num_requests is not None:
            launcher_entry['num_requests'] = arguments.num_requests

        return launcher_entry

    if launcher_entry['framework'].lower() != 'dlsdk':
        return launcher_entry

    launcher_entry.update(update_launcher_entry)
    _convert_models_args(launcher_entry)
    _async_evaluation_args(launcher_entry)
    _fpga_specific_args(launcher_entry)

    if 'cpu_extensions' not in launcher_entry and 'extensions' in arguments and arguments.extensions:
        extensions = arguments.extensions
        if not extensions.is_dir() or extensions.name == 'AUTO':
            launcher_entry['cpu_extensions'] = arguments.extensions

    if 'affinity_map' not in launcher_entry and 'affinity_map' in arguments and arguments.affinity_map:
        am = arguments.affinity_map
        if not am.is_dir():
            launcher_entry['affinity_map'] = arguments.affinity_map

    return launcher_entry
