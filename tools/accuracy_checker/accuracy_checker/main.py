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

import sys
from pathlib import Path
from argparse import ArgumentParser
from functools import partial
from csv import DictWriter

import cv2

from .config import ConfigReader
from .logging import print_info, add_file_handler, exception
from .evaluators import ModelEvaluator, PipeLineEvaluator, ModuleEvaluator
from .progress_reporters import ProgressReporter
from .utils import get_path, cast_to_bool, check_file_existence
from . import __version__

EVALUATION_MODE = {
    'models': ModelEvaluator,
    'pipelines': PipeLineEvaluator,
    'evaluations': ModuleEvaluator
}


def build_arguments_parser():
    parser = ArgumentParser(description='Deep Learning accuracy validation framework', allow_abbrev=False)
    parser.add_argument(
        '-d', '--definitions',
        help='path to the yml file with definitions',
        type=get_path,
        required=False
    )
    parser.add_argument(
        '-c', '--config',
        help='path to the yml file with local configuration',
        type=get_path,
        required=True
    )
    parser.add_argument(
        '-m', '--models',
        help='prefix path to the models and weights',
        type=partial(get_path, is_directory=True),
        required=False,
        nargs='+'
    )
    parser.add_argument(
        '-s', '--source',
        help='prefix path to the data source',
        type=partial(get_path, is_directory=True),
        required=False
    )
    parser.add_argument(
        '-a', '--annotations',
        help='prefix path to the converted annotations and datasets meta data',
        type=partial(get_path, is_directory=True),
        required=False
    )
    parser.add_argument(
        '-e', '--extensions',
        help='prefix path to extensions folder',
        type=partial(get_path, check_exists=False),
        default=Path.cwd(),
        required=False
    )
    parser.add_argument(
        '--cpu_extensions_mode',
        help='specified preferable set of processor instruction for automatic searching cpu extension lib',
        required=False,
        choices=['avx512', 'avx2', 'sse4']
    )
    parser.add_argument(
        '-b', '--bitstreams',
        help='prefix path to bitstreams folder',
        type=partial(get_path, file_or_directory=True),
        required=False
    )
    parser.add_argument(
        '--stored_predictions',
        help='path to file with saved predictions. Used for development',
        # since at the first time file does not exist and then created we can not always check existence
        required=False
    )
    parser.add_argument(
        '-C', '--converted_models',
        help='directory to store Model Optimizer converted models. Used for DLSDK launcher only',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    parser.add_argument(
        '-M', '--model_optimizer',
        help='path to model optimizer directory',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific locations
        # defined in model_conversion.py
        required=False
    )
    parser.add_argument(
        '--tf_custom_op_config_dir',
        help='path to directory with tensorflow custom operation configuration files for model optimizer',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific location
        # defined in model_conversion.py
        required=False
    )
    parser.add_argument(
        '--transformations_config_dir',
        help='path to directory with Model Optimizer transformations configuration files',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific location
        # defined in model_conversion.py
        required=False
    )
    parser.add_argument(
        '--tf_obj_detection_api_pipeline_config_path',
        help='path to directory with tensorflow object detection api pipeline configuration files for model optimizer',
        type=partial(get_path, is_directory=True),
        # there is no default value because if user did not specify it we use specific location
        # defined in model_conversion.py
        required=False
    )
    parser.add_argument(
        '--progress',
        help='progress reporter. You can select bar or print',
        required=False,
        default='bar'
    )
    parser.add_argument(
        '--progress_interval',
        help='interval for update progress if selected *print* progress.',
        required=False,
        type=int,
        default=1000
    )
    parser.add_argument(
        '-tf', '--target_framework',
        help='framework for infer',
        required=False
    )
    parser.add_argument(
        '-td', '--target_devices',
        help='Space separated list of devices for infer',
        required=False,
        nargs='+'
    )

    parser.add_argument(
        '-tt', '--target_tags',
        help='Space separated list of launcher tags for infer',
        required=False,
        nargs='+'
    )

    parser.add_argument(
        '-l', '--log_file',
        help='file for additional logging results',
        required=False
    )

    parser.add_argument(
        '--ignore_result_formatting',
        help='allow to get raw metrics results without data formatting',
        required=False,
        default=False,
        type=cast_to_bool
    )

    parser.add_argument(
        '-am', '--affinity_map',
        help='prefix path to the affinity maps',
        type=partial(get_path, file_or_directory=True),
        default=Path.cwd(),
        required=False
    )

    parser.add_argument(
        '--aocl',
        help='path to aocl executable for FPGA bitstream programming',
        type=get_path,
        required=False
    )
    parser.add_argument(
        '--vpu_log_level',
        help='log level for VPU devices',
        required=False,
        choices=['LOG_NONE', 'LOG_WARNING', 'LOG_INFO', 'LOG_DEBUG'],
        default='LOG_WARNING'
    )
    parser.add_argument(
        '--deprecated_ir_v7',
        help='Allow generation IR v7 via Model Optimizer',
        required=False,
        default=False,
        type=cast_to_bool
    )
    parser.add_argument(
        '-dc', '--device_config',
        help='Inference Engine device specific config file',
        type=get_path,
        required=False
    )

    parser.add_argument(
        '--async_mode',
        help='Allow evaluation in async mode',
        required=False,
        default=False,
        type=cast_to_bool
    )
    parser.add_argument(
        '--num_requests',
        help='the number of infer requests',
        required=False,
    )
    parser.add_argument(
        '--csv_result',
        help='file for results writing',
        required=False,
    )
    parser.add_argument(
        '--model_is_blob', help='the tip for automatic model search to use blob for dlsdk launcher',
        required=False,
        type=cast_to_bool
    )
    parser.add_argument(
        '--model_attributes', help="path's prefix for additional models attributes",
        required=False,
        type=partial(get_path, is_directory=True)
    )
    parser.add_argument(
        '-ss', '--subsample_size', help="dataset subsample size",
        required=False,
        type=str
    )
    parser.add_argument(
        '--shuffle', help="Allow shuffle annotation during creation a subset",
        required=False,
        type=cast_to_bool
    )
    parser.add_argument(
        '--version', action='version', version='%(prog)s {version}'.format(version=__version__),
        help='show tool version and exit'
    )

    return parser


def main():
    return_code = 0
    args = build_arguments_parser().parse_args()
    progress_bar_provider = args.progress if ':' not in args.progress else args.progress.split(':')[0]
    progress_reporter = ProgressReporter.provide(progress_bar_provider, None, print_interval=args.progress_interval)
    if args.log_file:
        add_file_handler(args.log_file)

    config, mode = ConfigReader.merge(args)
    evaluator_class = EVALUATION_MODE.get(mode)
    if not evaluator_class:
        raise ValueError('Unknown evaluation mode')
    for config_entry in config[mode]:
        try:
            processing_info = evaluator_class.get_processing_info(config_entry)
            print_processing_info(*processing_info)
            evaluator = evaluator_class.from_configs(config_entry)
            evaluator.process_dataset(stored_predictions=args.stored_predictions, progress_reporter=progress_reporter)
            metrics_results, _ = evaluator.extract_metrics_results(
                print_results=True, ignore_results_formatting=args.ignore_result_formatting
            )
            if args.csv_result:
                write_csv_result(args.csv_result, processing_info, metrics_results)
            evaluator.release()
        except Exception as e:  # pylint:disable=W0703
            exception(e)
            return_code = 1
            continue
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


def write_csv_result(csv_file, processing_info, metric_results):
    new_file = not check_file_existence(csv_file)
    field_names = ['model', 'launcher', 'device', 'dataset', 'tags', 'metric_name', 'metric_type', 'metric_value']
    model, launcher, device, tags, dataset = processing_info
    main_info = {
        'model': model,
        'launcher': launcher,
        'device': device.upper(),
        'tags': ' '.join(tags) if tags else '',
        'dataset': dataset
    }

    with open(csv_file, 'a+', newline='') as f:
        writer = DictWriter(f, fieldnames=field_names)
        if new_file:
            writer.writeheader()
        for metric_result in metric_results:
            writer.writerow({
                **main_info,
                'metric_name': metric_result['name'],
                'metric_type': metric_result['type'],
                'metric_value': metric_result['value']
            })


if __name__ == '__main__':
    main()
