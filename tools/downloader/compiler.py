#!/usr/bin/env python3

# Copyright (c) 2021 Intel Corporation
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
import sys
import common
from pathlib import Path

KNOWN_COMPILABLE_PRECISIONS = {'FP16', 'FP32'}

def compile(reporter, compiler_path, model, model_precision, args, output_dir):
    (output_dir / model.subdirectory).mkdir(parents=True, exist_ok=True)

    compile_cmd = [str(compiler_path),
        '-m={}'.format(args.model_dir / model.subdirectory / model_precision / (model.name + '.xml')),
        '-d={}'.format(args.target_device),
        '-ip={}'.format('U8' if args.input_precision is None else args.input_precision),
        '-op={}'.format('FP32' if args.output_precision is None else args.output_precision),
        '-o={}'.format(output_dir / model.subdirectory / model_precision / (model.name + '.blob')),
    ]
    reporter.print_section_heading('{}Compiling {} to BLOB ({})',
        '(DRY RUN) ' if args.dry_run else '', model.name, model_precision)

    reporter.print('Conversion command: {}', common.command_string(compile_cmd))
    if not args.dry_run:
        reporter.print(flush=True)
        if not reporter.job_context.subprocess(compile_cmd):
            return False
    reporter.print()
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='root of the directory tree with IR model files')
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        help='root of the directory tree to place compiled models files into')
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='compile only models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='compile only models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--input_precision', dest='input_precision',
        help='Input precision of compiled network')
    parser.add_argument('--output_precision', dest='output_precision',
        help='output_precision of compiled network')
    parser.add_argument('--precisions', metavar='PREC[,PREC...]',
        help='compile only specified precisions')
    parser.add_argument('--target_device', help='target device for the compiled model', default='MYRIAD')
    parser.add_argument('--all', action='store_true', help='compile all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    parser.add_argument('--compile_tool', type=Path, help='Compile Tool executable entry point')
    parser.add_argument('--dry_run', action='store_true',
        help='print the compilation commands without running them')
    args = parser.parse_args()


    compiler_path = args.compile_tool
    if compiler_path is None:
        try:
            compiler_path  = Path(os.environ['INTEL_OPENVINO_DIR']) / 'deployment_tools/tools/compile_tool/compile_tool'
        except KeyError:
            sys.exit('Unable to locate Compile Tool. '
                + 'Use --compiler or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')

    models = common.load_models_from_args(parser, args)

    if args.precisions is None:
        requested_precisions = KNOWN_COMPILABLE_PRECISIONS
    else:
        requested_precisions = set(args.precisions.split(','))
        unknown_precisions = requested_precisions - KNOWN_COMPILABLE_PRECISIONS
        if unknown_precisions:
            sys.exit('Unknown precisions specified: {}.'.format(', '.join(sorted(unknown_precisions))))

    reporter = common.Reporter(common.DirectOutputContext())

    output_dir = args.model_dir if args.output_dir is None else args.output_dir

    requested_precisions = KNOWN_COMPILABLE_PRECISIONS

    failed_models = []

    for model in models:
        if not model.compilable:
            reporter.print_section_heading('Skipping {} (compilation not supported)', model.name)
            reporter.print()
            continue
        for precision in sorted(requested_precisions):
            if not compile(reporter, compiler_path, model, precision, args, output_dir):
                failed_models.append(f'{model.name} ({precision})')
                continue

    if failed_models:
        reporter.print('FAILED:')
        for failed_model_name in failed_models:
            reporter.print(failed_model_name)
        sys.exit(1)

if __name__ == '__main__':
    main()
