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

from pathlib import Path
from argparse import ArgumentParser
from functools import partial

from .config import ConfigReader
from .logging import print_info, add_file_handler
from .model_evaluator import ModelEvaluator
from .progress_reporters import ProgressReporter
from .utils import get_path


def build_arguments_parser():
    parser = ArgumentParser(description='NN Validation on Caffe and IE')
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
        default=Path('config.yml'),
    )
    parser.add_argument(
        '-m', '--models',
        help='prefix path to the models and weights',
        type=partial(get_path, is_directory=True),
        default=Path.cwd(),
        required=False
    )
    parser.add_argument(
        '-s', '--source',
        help='prefix path to the data source',
        type=partial(get_path, is_directory=True),
        default=Path('datasets'),
        required=False
    )
    parser.add_argument(
        '-a', '--annotations',
        help='prefix path to the converted annotations and datasets meta data',
        type=partial(get_path, is_directory=True),
        default=Path('data/annotations'),
        required=False
    )
    parser.add_argument(
        '-e', '--extensions',
        help='prefix path to extensions folder',
        type=partial(get_path, is_directory=True),
        default=Path('extensions'),
        required=False
    )
    parser.add_argument(
        '-b', '--bitstreams',
        help='prefix path to bitstreams folder',
        type=partial(get_path, is_directory=True),
        default=Path('bitstreams'),
        required=False
    )
    parser.add_argument(
        '--stored_predictions',
        help='path to file with saved predictions. Used for development',
        # since at the first time file does not exist and then created we can not always check existence
        # type=get_path,
        required=False
    )
    parser.add_argument(
        '-C', '--converted_models',
        help='directory to store Model Optimizer converted models. Used for DLSDK launcher only',
        type=partial(get_path, is_directory=True),
        required=False
    )
    parser.add_argument(
        '-M', '--model_optimizer',
        help='path to model optimizer caffe directory',
        type=partial(get_path, is_directory=True),
        required=False
    )
    parser.add_argument(
        '--tf_custom_op_config',
        help='path to directory with tensorflow custom operation configuration files for model optimizer',
        type=partial(get_path, is_directory=True),
        required=False
    )
    parser.add_argument(
        '--progress',
        help='progress reporter',
        required=False,
        default='bar'
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
        nargs='*'
    )
    parser.add_argument(
        '-l', '--log_file',
        help='file for additional logging results',
        required=False
    )

    return parser


def main():
    args = build_arguments_parser().parse_args()
    progress_reporter = ProgressReporter.provide((
        args.progress if ':' not in args.progress
        else args.progress.split(':')[0]
    ))
    if args.log_file is not None:
        add_file_handler(args.log_file)

    config = ConfigReader.merge(args)

    for model in config['models']:
        for launcher_config in model['launchers']:
            for dataset_config in model['datasets']:
                print_processing_info(
                    model['name'],
                    launcher_config['framework'],
                    launcher_config['device'],
                    dataset_config['name']
                )
                model_evaluator = ModelEvaluator.from_configs(launcher_config, dataset_config)
                progress_reporter.reset(len(model_evaluator.dataset))
                model_evaluator.process_dataset(args.stored_predictions, progress_reporter=progress_reporter)
                model_evaluator.compute_metrics()

                model_evaluator.release()


def print_processing_info(model, launcher, device, dataset):
    print_info('Processing info:')
    print_info('model: {}'.format(model))
    print_info('launcher: {}'.format(launcher))
    print_info('device: {}'.format(device))
    print_info('dataset: {}'.format(dataset))


if __name__ == '__main__':
    main()
