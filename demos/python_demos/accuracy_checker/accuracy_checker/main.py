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
import pathlib
from argparse import ArgumentParser

from .config import ConfigReader
from .logging import print_info
from .model_evaluator import ModelEvaluator
from .progress_reporters import ProgressReporter
from .utils import check_exists


def build_arguments_parser():
    parser = ArgumentParser(description='NN Validation')
    parser.add_argument(
        '-r', '--root',
        help='prefix for all relative paths',
        type=check_exists,
        default=pathlib.Path.cwd(),
        required=False
    )
    parser.add_argument(
        '-d', '--definitions',
        help='path to the yml file with definitions',
        type=check_exists,
        required=False
    )
    parser.add_argument(
        '-c', '--config',
        help='path to the yml file with local configuration',
        type=check_exists,
        default=pathlib.Path('config.yml'),
    )
    parser.add_argument(
        '-m', '--models',
        help='prefix path to the models and weights',
        type=check_exists,
        default=pathlib.Path.cwd(),
        required=False
    )
    parser.add_argument(
        '-s', '--source',
        help='prefix path to the data source',
        type=pathlib.Path,
        default=pathlib.Path('datasets'),
        required=False
    )
    parser.add_argument(
        '-a', '--annotations',
        help='prefix path to the converted annotations and datasets meta data',
        type=pathlib.Path,
        default=pathlib.Path('data/annotations'),
        required=False
    )
    parser.add_argument(
        '-e', '--extensions',
        help='prefix path to extensions folder',
        type=pathlib.Path,
        default=pathlib.Path('extensions'),
        required=False
    )
    parser.add_argument(
        '-b', '--bitstreams',
        help='prefix path to bitstreams folder',
        type=pathlib.Path,
        default=pathlib.Path('bitstreams'),
        required=False
    )
    parser.add_argument(
        '--stored_predictions',
        help='path to file with saved predictions. Used for development',
        required=False
    )
    parser.add_argument(
        '-C', '--converted_models',
        help='directory to store Model Optimizer converted models. Used for DLSDK launcher only',
        type=check_exists,
        required=False
    )
    parser.add_argument(
        '-M', '--model_optimizer',
        help='path to Model Optimizer directory',
        type=check_exists,
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
        '-td', '--target_device',
        help='device for infer',
        required=False
    )

    return parser


def main():
    args = build_arguments_parser().parse_args()
    progress_reporter = ProgressReporter.provide((
        args.progress if ':' not in args.progress
        else args.progress.split(':')[0]
    ))

    config = ConfigReader.merge(args)

    for model in config['models']:
        for launcher_config in model['launchers']:
            for dataset_config in model['datasets']:
                model_evaluator = ModelEvaluator.from_configs(launcher_config, dataset_config)

                print_info("Processing: {}".format(model_evaluator.dataset.name))
                progress_reporter.reset(len(model_evaluator.dataset))
                model_evaluator.process_dataset(args.stored_predictions, progress_reporter=progress_reporter)
                model_evaluator.compute_metrics()

                model_evaluator.release()


if __name__ == '__main__':
    main()
