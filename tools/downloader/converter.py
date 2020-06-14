#!/usr/bin/env python3

# Copyright (c) 2019 Intel Corporation
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
import os
import re
import string
import sys

from pathlib import Path

import common


def convert_to_onnx(reporter, model, output_dir, args):
    reporter.print_section_heading('{}Converting {} to ONNX',
        '(DRY RUN) ' if args.dry_run else '', model.name)

    conversion_to_onnx_args = [string.Template(arg).substitute(conv_dir=output_dir / model.subdirectory,
                                                               dl_dir=args.download_dir / model.subdirectory)
                               for arg in model.conversion_to_onnx_args]
    cmd = [str(args.python), str(Path(__file__).absolute().parent / model.converter_to_onnx), *conversion_to_onnx_args]

    reporter.print('Conversion to ONNX command: {}', common.command_string(cmd))
    reporter.print(flush=True)

    success = True if args.dry_run else reporter.job_context.subprocess(cmd)
    reporter.print()

    return success

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

    mo_path = args.mo
    if mo_path is None:
        try:
            mo_path = Path(os.environ['INTEL_OPENVINO_DIR']) / 'deployment_tools/model_optimizer/mo.py'
        except KeyError:
            sys.exit('Unable to locate Model Optimizer. '
                + 'Use --mo or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')

    extra_mo_args = args.extra_mo_args or []

    if args.precisions is None:
        requested_precisions = common.KNOWN_PRECISIONS
    else:
        requested_precisions = set(args.precisions.split(','))
        unknown_precisions = requested_precisions - common.KNOWN_PRECISIONS
        if unknown_precisions:
            sys.exit('Unknown precisions specified: {}.'.format(', '.join(sorted(unknown_precisions))))

    models = common.load_models_from_args(parser, args)

    output_dir = args.download_dir if args.output_dir is None else args.output_dir

    def convert(reporter, model):
        if model.mo_args is None:
            reporter.print_section_heading('Skipping {} (no conversions defined)', model.name)
            reporter.print()
            return True

        model_precisions = requested_precisions & model.precisions
        if not model_precisions:
            reporter.print_section_heading('Skipping {} (all conversions skipped)', model.name)
            reporter.print()
            return True

        model_format = model.framework

        if model.conversion_to_onnx_args:
            if not convert_to_onnx(reporter, model, output_dir, args):
                return False
            model_format = 'onnx'

        expanded_mo_args = [
            string.Template(arg).substitute(dl_dir=args.download_dir / model.subdirectory,
                                            mo_dir=mo_path.parent,
                                            conv_dir=output_dir / model.subdirectory,
                                            config_dir=common.MODEL_ROOT / model.subdirectory)
            for arg in model.mo_args]

        for model_precision in sorted(model_precisions):
            mo_cmd = [str(args.python), '--', str(mo_path),
                '--framework={}'.format(model_format),
                '--data_type={}'.format(model_precision),
                '--output_dir={}'.format(output_dir / model.subdirectory / model_precision),
                '--model_name={}'.format(model.name),
                *expanded_mo_args, *extra_mo_args]

            reporter.print_section_heading('{}Converting {} to IR ({})',
                '(DRY RUN) ' if args.dry_run else '', model.name, model_precision)

            reporter.print('Conversion command: {}', common.command_string(mo_cmd))

            if not args.dry_run:
                reporter.print(flush=True)

                if not reporter.job_context.subprocess(mo_cmd):
                    return False

            reporter.print()

        return True

    reporter = common.Reporter(common.DirectOutputContext())

    if args.jobs == 1 or args.dry_run:
        results = [convert(reporter, model) for model in models]
    else:
        results = common.run_in_parallel(args.jobs,
            lambda context, model: convert(common.Reporter(context), model),
            models)

    failed_models = [model.name for model, successful in zip(models, results) if not successful]

    if failed_models:
        reporter.print('FAILED:')
        for failed_model_name in failed_models:
            reporter.print(failed_model_name)
        sys.exit(1)

if __name__ == '__main__':
    main()
