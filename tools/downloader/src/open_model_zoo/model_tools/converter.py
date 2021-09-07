# Copyright (c) 2019-2021 Intel Corporation
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
import collections
import json
import os
import string
import sys

from pathlib import Path

from open_model_zoo.model_tools import (
    _configuration, _common, _concurrency, _reporting,
)

ModelOptimizerProperties = collections.namedtuple('ModelOptimizerProperties',
    ['cmd_prefix', 'extra_args', 'base_dir'])

def run_pre_convert(reporter, model, output_dir, args):
    script = _common.MODEL_ROOT / model.subdirectory / 'pre-convert.py'
    if not script.exists():
        return True

    reporter.print_section_heading('{}Running pre-convert script for {}',
        '(DRY RUN) ' if args.dry_run else '', model.name)

    cmd = [str(args.python), '--', str(script), '--',
        str(args.download_dir / model.subdirectory), str(output_dir / model.subdirectory)]

    reporter.print('Pre-convert command: {}', _common.command_string(cmd))
    reporter.print(flush=True)

    success = True if args.dry_run else reporter.job_context.subprocess(cmd)
    reporter.print()

    return success

def convert_to_onnx(reporter, model, output_dir, args, template_variables):
    reporter.print_section_heading('{}Converting {} to ONNX',
        '(DRY RUN) ' if args.dry_run else '', model.name)

    converter_path = Path(__file__).absolute().parent / \
        'internal_scripts' / model.converter_to_onnx

    conversion_to_onnx_args = [string.Template(arg).substitute(template_variables)
                               for arg in model.conversion_to_onnx_args]

    cmd = [str(args.python), '--', str(converter_path), *conversion_to_onnx_args]

    reporter.print('Conversion to ONNX command: {}', _common.command_string(cmd))
    reporter.print(flush=True)

    success = True if args.dry_run else reporter.job_context.subprocess(cmd)
    reporter.print()

    return success

def convert(reporter, model, output_dir, args, mo_props, requested_precisions):
    telemetry = _common.Telemetry()
    if model.mo_args is None:
        reporter.print_section_heading('Skipping {} (no conversions defined)', model.name)
        reporter.print()
        return True

    model_precisions = requested_precisions & model.precisions
    if not model_precisions:
        reporter.print_section_heading('Skipping {} (all conversions skipped)', model.name)
        reporter.print()
        return True

    (output_dir / model.subdirectory).mkdir(parents=True, exist_ok=True)

    if not run_pre_convert(reporter, model, output_dir, args):
        telemetry.send_event('md', 'converter_failed_models', model.name)
        telemetry.send_event('md', 'converter_error',
            json.dumps({'error': 'pre-convert-script-failed', 'model': model.name, 'precision': None}))
        return False

    model_format = model.framework

    template_variables = {
        'config_dir': _common.MODEL_ROOT / model.subdirectory,
        'conv_dir': output_dir / model.subdirectory,
        'dl_dir': args.download_dir / model.subdirectory,
        'mo_dir': mo_props.base_dir,
    }

    if model.conversion_to_onnx_args:
        if not convert_to_onnx(reporter, model, output_dir, args, template_variables):
            telemetry.send_event('md', 'converter_failed_models', model.name)
            telemetry.send_event('md', 'converter_error',
                json.dumps({'error': 'convert_to_onnx-failed', 'model': model.name, 'precision': None}))
            return False
        model_format = 'onnx'

    expanded_mo_args = [
        string.Template(arg).substitute(template_variables)
        for arg in model.mo_args]

    for model_precision in sorted(model_precisions):
        data_type = model_precision.split('-')[0]
        mo_cmd = [*mo_props.cmd_prefix,
            '--framework={}'.format(model_format),
            '--data_type={}'.format(data_type),
            '--output_dir={}'.format(output_dir / model.subdirectory / model_precision),
            '--model_name={}'.format(model.name),
            *expanded_mo_args, *mo_props.extra_args]

        reporter.print_section_heading('{}Converting {} to IR ({})',
            '(DRY RUN) ' if args.dry_run else '', model.name, model_precision)

        reporter.print('Conversion command: {}', _common.command_string(mo_cmd))

        if not args.dry_run:
            reporter.print(flush=True)

            if not reporter.job_context.subprocess(mo_cmd):
                telemetry.send_event('md', 'converter_failed_models', model.name)
                telemetry.send_event('md', 'converter_error',
                    json.dumps({'error': 'mo-failed', 'model': model.name, 'precision': model_precision}))
                return False

        reporter.print()

    return True

def num_jobs_arg(value_str):
    if value_str == 'auto':
        return os.cpu_count() or 1

    try:
        value = int(value_str)
        if value > 0: return value
    except ValueError:
        pass

    raise argparse.ArgumentTypeError('must be a positive integer or "auto" (got {!r})'.format(value_str))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='root of the directory tree with downloaded model files')
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        help='root of the directory tree to place converted files into')
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='convert only models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='convert only models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--all', action='store_true', help='convert all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    parser.add_argument('--precisions', metavar='PREC[,PREC...]',
        help='run only conversions that produce models with the specified precisions')
    parser.add_argument('-p', '--python', type=Path, metavar='PYTHON', default=sys.executable,
        help='Python executable to run Model Optimizer with')
    parser.add_argument('--mo', type=Path, metavar='MO.PY',
        help='Model Optimizer entry point script')
    parser.add_argument('--add_mo_arg', dest='extra_mo_args', metavar='ARG', action='append',
        help='Extra argument to pass to Model Optimizer')
    parser.add_argument('--dry_run', action='store_true',
        help='Print the conversion commands without running them')
    parser.add_argument('-j', '--jobs', type=num_jobs_arg, default=1,
        help='number of conversions to run concurrently')

    # aliases for backwards compatibility
    parser.add_argument('--add-mo-arg', dest='extra_mo_args', action='append', help=argparse.SUPPRESS)
    parser.add_argument('--dry-run', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()

    with _common.telemetry_session('Model Converter', 'converter') as telemetry:
        models = _configuration.load_models_from_args(parser, args)
        for mode in ['all', 'list', 'name']:
            if getattr(args, mode):
                telemetry.send_event('md', 'converter_selection_mode', mode)

        if args.precisions is None:
            requested_precisions = _common.KNOWN_PRECISIONS
        else:
            requested_precisions = set(args.precisions.split(','))

        for model in models:
            precisions_to_send = requested_precisions if args.precisions else requested_precisions & model.precisions
            model_information = {
                'name': model.name,
                'framework': model.framework,
                'precisions': str(precisions_to_send).replace(',', ';'),
            }
            telemetry.send_event('md', 'converter_model', json.dumps(model_information))

        unknown_precisions = requested_precisions - _common.KNOWN_PRECISIONS
        if unknown_precisions:
            sys.exit('Unknown precisions specified: {}.'.format(', '.join(sorted(unknown_precisions))))

        mo_path = args.mo

        if mo_path is None:
            mo_package_path = _common.get_package_path(args.python, 'mo')

            if mo_package_path:
                # run MO as a module
                mo_cmd_prefix = [str(args.python), '-m', 'mo']
                mo_dir = mo_package_path.parent
            else:
                try:
                    mo_path = Path(os.environ['INTEL_OPENVINO_DIR']) / 'deployment_tools/model_optimizer/mo.py'
                except KeyError:
                    sys.exit('Unable to locate Model Optimizer. '
                        + 'Use --mo or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')

        if mo_path is not None:
            # run MO as a script
            mo_cmd_prefix = [str(args.python), '--', str(mo_path)]
            mo_dir = mo_path.parent

        output_dir = args.download_dir if args.output_dir is None else args.output_dir

        reporter = _reporting.Reporter(_reporting.DirectOutputContext())
        mo_props = ModelOptimizerProperties(
            cmd_prefix=mo_cmd_prefix,
            extra_args=args.extra_mo_args or [],
            base_dir=mo_dir,
        )
        shared_convert_args = (output_dir, args, mo_props, requested_precisions)

        if args.jobs == 1 or args.dry_run:
            results = [convert(reporter, model, *shared_convert_args) for model in models]
        else:
            results = _concurrency.run_in_parallel(args.jobs,
                lambda context, model:
                    convert(_reporting.Reporter(context), model, *shared_convert_args),
                models)

        failed_models = [model.name for model, successful in zip(models, results) if not successful]

        if failed_models:
            reporter.print('FAILED:')
            for failed_model_name in failed_models:
                reporter.print(failed_model_name)
            sys.exit(1)

if __name__ == '__main__':
    main()
