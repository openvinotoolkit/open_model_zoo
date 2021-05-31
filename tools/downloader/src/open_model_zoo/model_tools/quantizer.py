# Copyright (c) 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import sys
import tempfile

from pathlib import Path

import yaml

from open_model_zoo.model_tools import (
    _configuration, _common, _reporting,
)


DEFAULT_POT_CONFIG_BASE = {
    'compression': {
        'algorithms': [
            {
                'name': 'DefaultQuantization',
                'params': {
                    'preset': 'performance',
                    'stat_subset_size': 300,
                },
            },
        ],
    },
}

def quantize(reporter, model, precision, args, output_dir, pot_cmd_prefix, pot_env):
    input_precision = _common.KNOWN_QUANTIZED_PRECISIONS[precision]

    pot_config_base_path = _common.MODEL_ROOT / model.subdirectory / 'quantization.yml'

    try:
        with pot_config_base_path.open('rb') as pot_config_base_file:
            pot_config_base = yaml.safe_load(pot_config_base_file)
    except FileNotFoundError:
        pot_config_base = DEFAULT_POT_CONFIG_BASE

    pot_config_paths = {
        'engine': {
            'config': str(_common.MODEL_ROOT/ model.subdirectory / 'accuracy-check.yml'),
        },
        'model': {
            'model': str(args.model_dir / model.subdirectory / input_precision / (model.name + '.xml')),
            'weights': str(args.model_dir / model.subdirectory / input_precision / (model.name + '.bin')),
            'model_name': model.name,
        }
    }

    pot_config = {**pot_config_base, **pot_config_paths}

    if args.target_device:
        pot_config['compression']['target_device'] = args.target_device

    reporter.print_section_heading('{}Quantizing {} from {} to {}',
        '(DRY RUN) ' if args.dry_run else '', model.name, input_precision, precision)

    model_output_dir = output_dir / model.subdirectory / precision
    pot_config_path = model_output_dir / 'pot-config.json'

    reporter.print('Creating {}...', pot_config_path)
    pot_config_path.parent.mkdir(parents=True, exist_ok=True)
    with pot_config_path.open('w') as pot_config_file:
        json.dump(pot_config, pot_config_file, indent=4)
        pot_config_file.write('\n')

    pot_output_dir = model_output_dir / 'pot-output'
    pot_output_dir.mkdir(parents=True, exist_ok=True)

    pot_cmd = [*pot_cmd_prefix,
        '--config={}'.format(pot_config_path),
        '--direct-dump',
        '--output-dir={}'.format(pot_output_dir),
    ]

    reporter.print('Quantization command: {}', _common.command_string(pot_cmd))
    reporter.print('Quantization environment: {}',
        ' '.join('{}={}'.format(k, _common.quote_arg(v))
            for k, v in sorted(pot_env.items())))

    success = True

    if not args.dry_run:
        reporter.print(flush=True)

        success = reporter.job_context.subprocess(pot_cmd, env={**os.environ, **pot_env})

    reporter.print()
    if not success: return False

    if not args.dry_run:
        reporter.print('Moving quantized model to {}...', model_output_dir)
        for ext in ['.xml', '.bin']:
            (pot_output_dir / 'optimized' / (model.name + ext)).replace(
                model_output_dir / (model.name + ext))
        reporter.print()

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='root of the directory tree with the full precision model files')
    parser.add_argument('--dataset_dir', type=Path, help='root of the dataset directory tree')
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        help='root of the directory tree to place quantized model files into')
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='quantize only models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='quantize only models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all', action='store_true', help='quantize all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    parser.add_argument('-p', '--python', type=Path, metavar='PYTHON', default=sys.executable,
        help='Python executable to run Post-Training Optimization Toolkit with')
    parser.add_argument('--pot', type=Path, help='Post-Training Optimization Toolkit entry point script')
    parser.add_argument('--dry_run', action='store_true',
        help='print the quantization commands without running them')
    parser.add_argument('--precisions', metavar='PREC[,PREC...]',
        help='quantize only to the specified precisions')
    parser.add_argument('--target_device', help='target device for the quantized model')
    args = parser.parse_args()

    with _common.telemetry_session('Model Quantizer', 'quantizer') as telemetry:
        models = _configuration.load_models_from_args(parser, args)
        for mode in ['all', 'list', 'name']:
            if getattr(args, mode):
                telemetry.send_event('md', 'quantizer_selection_mode', mode)

        if args.precisions is None:
            requested_precisions = _common.KNOWN_QUANTIZED_PRECISIONS.keys()
        else:
            requested_precisions = set(args.precisions.split(','))

        for model in models:
            model_information = {
                'name': model.name,
                'framework': model.framework,
                'precisions': str(requested_precisions).replace(',', ';'),
            }
            telemetry.send_event('md', 'quantizer_model', json.dumps(model_information))

        unknown_precisions = requested_precisions - _common.KNOWN_QUANTIZED_PRECISIONS.keys()
        if unknown_precisions:
            sys.exit('Unknown precisions specified: {}.'.format(', '.join(sorted(unknown_precisions))))

        pot_path = args.pot
        if pot_path is None:
            if _common.get_package_path(args.python, 'pot'):
                # run POT as a module
                pot_cmd_prefix = [str(args.python), '-m', 'pot']
            else:
                try:
                    pot_path = Path(os.environ['INTEL_OPENVINO_DIR']) / 'deployment_tools/tools/post_training_optimization_toolkit/main.py'
                except KeyError:
                    sys.exit('Unable to locate Post-Training Optimization Toolkit. '
                        + 'Use --pot or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')

        if pot_path is not None:
            # run POT as a script
            pot_cmd_prefix = [str(args.python), '--', str(pot_path)]

        # We can't mark it as required, because it's not required when --print_all is specified.
        # So we have to check it manually.
        if not args.dataset_dir:
            sys.exit('--dataset_dir must be specified.')

        reporter = _reporting.Reporter(_reporting.DirectOutputContext())

        output_dir = args.output_dir or args.model_dir

        failed_models = []

        with tempfile.TemporaryDirectory() as temp_dir:
            annotation_dir = Path(temp_dir) / 'annotations'
            annotation_dir.mkdir()

            pot_env = {
                'ANNOTATIONS_DIR': str(annotation_dir),
                'DATA_DIR': str(args.dataset_dir),
                'DEFINITIONS_FILE': str(_common.DATASET_DEFINITIONS),
            }

            for model in models:
                if not model.quantization_output_precisions:
                    reporter.print_section_heading('Skipping {} (quantization not supported)', model.name)
                    reporter.print()
                    continue

                model_precisions = requested_precisions & model.quantization_output_precisions

                if not model_precisions:
                    reporter.print_section_heading('Skipping {} (all precisions skipped)', model.name)
                    reporter.print()
                    continue

                pot_env.update({
                    'MODELS_DIR': str(args.model_dir / model.subdirectory)
                })

                for precision in sorted(model_precisions):
                    if not quantize(reporter, model, precision, args, output_dir, pot_cmd_prefix, pot_env):
                        failed_models.append(model.name)
                        break


        if failed_models:
            reporter.print('FAILED:')
            for failed_model_name in failed_models:
                reporter.print(failed_model_name)
            sys.exit(1)

if __name__ == '__main__':
    main()
