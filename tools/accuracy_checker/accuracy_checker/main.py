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

import json
import sys
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser
from functools import partial
from csv import DictWriter

import cv2

from .config import ConfigReader
from .logging import print_info, add_file_handler, exception
from .evaluators import ModelEvaluator, ModuleEvaluator
from .progress_reporters import ProgressReporter
from .utils import (
    get_path,
    cast_to_bool,
    check_file_existence,
    validate_print_interval,
    start_telemetry,
    end_telemetry,
    send_telemetry_event
)
from . import __version__


EVALUATION_MODE = {
    'models': ModelEvaluator,
    'evaluations': ModuleEvaluator
}


def add_common_args(parser):
    common_args = parser.add_argument_group('Common arguments')
    common_args.add_argument(
        '-d', '--definitions',
        help='path to the yml file with definitions',
        type=get_path,
        required=False
    )
    common_args.add_argument(
        '-c', '--config',
        help='path to the yml file with local configuration',
        type=get_path,
        required=True
    )
    common_args.add_argument(
        '-m', '--models',
        help='prefix path to the models and weights',
        type=partial(get_path, file_or_directory=True),
        required=False,
        nargs='+'
    )
    common_args.add_argument(
        '-s', '--source',
        help='prefix path to the data source',
        type=partial(get_path, is_directory=True),
        required=False
    )
    common_args.add_argument(
        '-a', '--annotations',
        help='prefix path to the converted annotations and datasets meta data',
        type=partial(get_path, is_directory=True),
        required=False
    )
    common_args.add_argument(
        '--model_attributes',
        help="path's prefix for additional models attributes",
        type=partial(get_path, is_directory=True),
        required=False
    )
    common_args.add_argument(
        '--input_precision',
        help='space-separated list of precisions for network inputs. '
             'Providing several values required <layer_name>:<precision> format. '
             'If single value without layer_name provided, then it will be applied to all input layers.',
        required=False,
        nargs='+'
    )


def add_config_filtration_args(parser):
    config_filtration_args = parser.add_argument_group('Config filtration arguments')
    config_filtration_args.add_argument(
        '-tf', '--target_framework',
        help='framework for infer',
        required=False
    )
    config_filtration_args.add_argument(
        '-td', '--target_devices',
        help='space separated list of devices for infer',
        required=False,
        nargs='+'
    )
    config_filtration_args.add_argument(
        '-tt', '--target_tags',
        help='space separated list of launcher tags for infer',
        required=False,
        nargs='+'
    )


def add_dataset_related_args(parser):
    dataset_related_args = parser.add_argument_group('Dataset related arguments')
    dataset_related_args.add_argument(
        '-ss', '--subsample_size',
        help="dataset subsample size",
        type=str,
        required=False
    )
    dataset_related_args.add_argument(
        '--shuffle',
        help="allow shuffle annotation during creation a subset",
        type=cast_to_bool,
        required=False
    )
    dataset_related_args.add_argument(
        '--store_subset',
        help='allow to save evaluation data ids',
        type=cast_to_bool,
        default=False,
        required=False
    )
    dataset_related_args.add_argument(
        '--subset_file',
        help='file name for saving or reading identifiers subset',
        required=False
    )


def add_profiling_related_args(parser):
    profiling_related_args = parser.add_argument_group('Profiling related arguments')
    profiling_related_args.add_argument(
        '--profile',
        help='activate metric profiling mode',
        type=cast_to_bool,
        required=False
    )
    profiling_related_args.add_argument(
        '--profiler_logs_dir',
        help='path to save profiler logs',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    profiling_related_args.add_argument(
        '--profile_report_type',
        help='report type for profiler logs',
        default='csv',
        choices=['csv', 'json'],
        required=False
    )


def add_tool_settings_args(parser):
    tool_settings_args = parser.add_argument_group('Tool settings arguments')
    tool_settings_args.add_argument(
        '--progress',
        help='progress reporter. You can select bar or print',
        default='bar',
        required=False
    )
    tool_settings_args.add_argument(
        '--progress_interval',
        help='interval for update progress if selected *print* progress.',
        type=int,
        default=1000,
        required=False
    )
    tool_settings_args.add_argument(
        '--ignore_result_formatting',
        help='allow to get raw metrics results without data formatting',
        type=cast_to_bool,
        default=False,
        required=False
    )
    tool_settings_args.add_argument(
        '--stored_predictions',
        help='path to file with saved predictions. Used for development',
        # since at the first time file does not exist and then created we can not always check existence
        required=False
    )
    tool_settings_args.add_argument(
        '--csv_result',
        help='file for results writing',
        required=False,
    )
    tool_settings_args.add_argument(
        '--intermediate_metrics_results',
        help='enables intermediate metrics results printing',
        type=cast_to_bool,
        default=False,
        required=False
    )
    tool_settings_args.add_argument(
        '--metrics_interval',
        help='number of iteration for updated metrics result printing',
        type=int,
        default=1000,
        required=False
    )
    tool_settings_args.add_argument(
        '--store_only',
        type=cast_to_bool,
        default=False,
        required=False
    )
    tool_settings_args.add_argument(
        '-l', '--log_file',
        help='file for additional logging results',
        required=False
    )


def add_openvino_specific_args(parser):
    openvino_specific_args = parser.add_argument_group('OpenVINO specific arguments')
    openvino_specific_args.add_argument(
        '-e', '--extensions',
        help='prefix path to extensions folder',
        type=partial(get_path, check_exists=False),
        default=Path.cwd(),
        required=False
    )
    openvino_specific_args.add_argument(
        '--cpu_extensions_mode',
        help='specified preferable set of processor instruction for automatic searching cpu extension lib',
        choices=['avx512', 'avx2', 'sse4'],
        required=False
    )
    openvino_specific_args.add_argument(
        '-b', '--bitstreams',
        help='prefix path to bitstreams folder',
        type=partial(get_path, file_or_directory=True),
        required=False
    )
    openvino_specific_args.add_argument(
        '-M', '--model_optimizer',
        help='path to model optimizer directory',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific locations
        # defined in model_conversion.py
        required=False
    )
    openvino_specific_args.add_argument(
        '--tf_custom_op_config_dir',
        help='path to directory with tensorflow custom operation configuration files for model optimizer',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific location
        # defined in model_conversion.py
        required=False
    )
    openvino_specific_args.add_argument(
        '--transformations_config_dir',
        help='path to directory with Model Optimizer transformations configuration files',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific location
        # defined in model_conversion.py
        required=False
    )
    openvino_specific_args.add_argument(
        '--tf_obj_detection_api_pipeline_config_path',
        help='path to directory with tensorflow object detection api pipeline configuration files for model optimizer',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific location
        # defined in model_conversion.py
        required=False
    )
    openvino_specific_args.add_argument(
        '--deprecated_ir_v7',
        help='allow generation IR v7 via Model Optimizer',
        type=cast_to_bool,
        default=False,
        required=False
    )
    openvino_specific_args.add_argument(
        '-dc', '--device_config',
        help='Inference Engine device specific config file',
        type=get_path,
        required=False
    )
    openvino_specific_args.add_argument(
        '--ie_preprocessing',
        help='enable preprocessing via Inference Engine. Accepted only for dlsdk launcher.',
        type=cast_to_bool,
        default=False,
        required=False
    )
    openvino_specific_args.add_argument(
        '--model_is_blob',
        help='the tip for automatic model search to use blob for dlsdk launcher',
        type=cast_to_bool,
        required=False
    )
    openvino_specific_args.add_argument(
        '-C', '--converted_models',
        help='directory to store Model Optimizer converted models. Used for DLSDK launcher only',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    openvino_specific_args.add_argument(
        '-am', '--affinity_map',
        help='prefix path to the affinity maps',
        type=partial(get_path, file_or_directory=True),
        default=Path.cwd(),
        required=False
    )
    openvino_specific_args.add_argument(
        '--aocl',
        help='path to aocl executable for FPGA bitstream programming',
        type=get_path,
        required=False
    )
    openvino_specific_args.add_argument(
        '--vpu_log_level',
        help='log level for VPU devices',
        default='LOG_WARNING',
        choices=['LOG_NONE', 'LOG_WARNING', 'LOG_INFO', 'LOG_DEBUG'],
        required=False
    )
    openvino_specific_args.add_argument(
        '--async_mode',
        help='Allow evaluation in async mode',
        type=cast_to_bool,
        default=False,
        required=False
    )
    openvino_specific_args.add_argument(
        '--num_requests',
        help='the number of infer requests',
        required=False
    )
    openvino_specific_args.add_argument(
        '--kaldi_bin_dir', help='directory with Kaldi utility binaries. Required only for Kaldi models decoding.',
        required=False, type=partial(get_path, is_directory=True)
    )
    openvino_specific_args.add_argument(
        '--kaldi_log_file', help='path for saving logs from Kaldi tools', type=partial(get_path, check_exists=False),
        required=False
    )


def build_arguments_parser():
    parser = ArgumentParser(description='Deep Learning accuracy validation framework', allow_abbrev=False)
    add_common_args(parser)
    add_config_filtration_args(parser)
    add_dataset_related_args(parser)
    add_profiling_related_args(parser)
    add_tool_settings_args(parser)
    add_openvino_specific_args(parser)

    parser.add_argument(
        '--version',
        help='show tool version and exit',
        action='version',
        version='%(prog)s {version}'.format(version=__version__)
    )

    return parser


def main():
    return_code = 0
    args = build_arguments_parser().parse_args()
    tm = start_telemetry()
    progress_bar_provider = args.progress if ':' not in args.progress else args.progress.split(':')[0]
    progress_reporter = ProgressReporter.provide(progress_bar_provider, None, print_interval=args.progress_interval)
    if args.log_file:
        add_file_handler(args.log_file)
    evaluator_kwargs = {}
    if args.intermediate_metrics_results:
        validate_print_interval(args.metrics_interval)
        evaluator_kwargs['intermediate_metrics_results'] = args.intermediate_metrics_results
        evaluator_kwargs['metrics_interval'] = args.metrics_interval
        evaluator_kwargs['ignore_result_formatting'] = args.ignore_result_formatting
    evaluator_kwargs['store_only'] = args.store_only
    details = {
        'mode': "online" if not args.store_only else "offline",
        'metric_profiling': args.profile,
        'error': None
    }

    config, mode = ConfigReader.merge(args)
    evaluator_class = EVALUATION_MODE.get(mode)
    if not evaluator_class:
        send_telemetry_event(tm, 'error', 'Unknown evaluation mode')
        end_telemetry(tm)
        raise ValueError('Unknown evaluation mode')
    for config_entry in config[mode]:
        details.update({'status': 'started', "error": None})
        config_entry.update({
            '_store_only': args.store_only,
            '_stored_data': args.stored_predictions
        })
        try:
            processing_info = evaluator_class.get_processing_info(config_entry)
            print_processing_info(*processing_info)
            evaluator = evaluator_class.from_configs(config_entry)
            details.update(evaluator.send_processing_info(tm))
            if args.profile:
                setup_profiling(args.profiler_log_dir, evaluator)
            send_telemetry_event(tm, 'model_run', details)
            evaluator.process_dataset(
                stored_predictions=args.stored_predictions, progress_reporter=progress_reporter, **evaluator_kwargs
            )
            if not args.store_only:
                metrics_results, metrics_meta = evaluator.extract_metrics_results(
                    print_results=True, ignore_results_formatting=args.ignore_result_formatting
                )
                if args.csv_result:
                    write_csv_result(
                        args.csv_result, processing_info, metrics_results, evaluator.dataset_size, metrics_meta
                    )
            evaluator.release()
            details['status'] = 'finished'
            send_telemetry_event(tm, 'model_run', details)

        except Exception as e:  # pylint:disable=W0703
            details['status'] = 'error'
            details['error'] = str(type(e))
            send_telemetry_event(tm, 'model_run', json.dumps(details))
            exception(e)
            return_code = 1
            continue
        end_telemetry(tm)
    sys.exit(return_code)


def print_processing_info(model, launcher, device, tags, dataset):
    print_info('Processing info:')
    print_info('model: {}'.format(model))
    print_info('launcher: {}'.format(launcher))
    if tags:
        print_info('launcher tags: {}'.format(' '.join(tags)))
    print_info('device: {}'.format(device.upper()))
    print_info('dataset: {}'.format(dataset))
    print_info('OpenCV version: {}'.format(cv2.__version__))


def write_csv_result(csv_file, processing_info, metric_results, dataset_size, metrics_meta):
    new_file = not check_file_existence(csv_file)
    field_names = [
        'model', 'launcher', 'device', 'dataset',
        'tags', 'metric_name', 'metric_type', 'metric_value', 'metric_target', 'metric_scale', 'metric_postfix',
        'dataset_size', 'ref', 'abs_threshold', 'rel_threshold']
    model, launcher, device, tags, dataset = processing_info
    main_info = {
        'model': model,
        'launcher': launcher,
        'device': device.upper(),
        'tags': ' '.join(tags) if tags else '',
        'dataset': dataset,
        'dataset_size': dataset_size
    }

    with open(csv_file, 'a+', newline='') as f:
        writer = DictWriter(f, fieldnames=field_names)
        if new_file:
            writer.writeheader()
        for metric_result, metric_meta in zip(metric_results, metrics_meta):
            writer.writerow({
                **main_info,
                'metric_name': metric_result['name'],
                'metric_type': metric_result['type'],
                'metric_value': metric_result['value'],
                'metric_target': metric_meta.get('target', 'higher-better'),
                'metric_scale': metric_meta.get('scale', 100),
                'metric_postfix': metric_meta.get('postfix', '%'),
                'ref': metric_result.get('ref', ''),
                'abs_threshold': metric_result.get('abs_threshold', 0),
                'rel_threshold': metric_result.get('rel_threshold', 0)
            })


def setup_profiling(logs_dir, evaluator):
    _timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    profiler_dir = logs_dir / _timestamp
    print_info('Metric profiling activated. Profiler output will be stored in {}'.format(profiler_dir))
    evaluator.set_profiling_dir(profiler_dir)


if __name__ == '__main__':
    main()
