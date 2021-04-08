#!/usr/bin/env python3

# Copyright (c) 2022 Intel Corporation
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
import yaml
import common
from pathlib import Path

def compile(compiler_path, model,target_device, output_dir):
    
    compile_cmd = [str(compiler_path), 
        '-d={}'.format(target_device),
        '-ip={}'.format(compiler_input_precision),
        '-op={}'.format(compiler_input_precision),
    ]
    
    success = True

    if not args.dry_run:
        reporter.print(flush=True)

        success = reporter.job_context.subprocess(ompile_cmd, env={**os.environ, **pot_env})
    reporter.print()
    return True

def main():
    parser.add_argument('--model_dir', type=Path, metavar='DIR',
        default=Path.cwd(), help='root of the directory tree with IR model files')
    parser.add_argument('-o', '--output_dir', type=Path, metavar='DIR',
        help='root of the directory tree to place compiled models files into')
    parser.add_argument('--name', metavar='PAT[,PAT...]',
        help='compile only models whose names match at least one of the specified patterns')
    parser.add_argument('--list', type=Path, metavar='FILE.LST',
        help='compile only models whose names match at least one of the patterns in the specified file')
    parser.add_argument('--target_device', help='target device for the compiled model')
    parser.add_argument('--all', action='store_true', help='quantize all available models')
    parser.add_argument('--print_all', action='store_true', help='print all available models')
    parser.add_argument('--ct', type=Path, help='Compiler Tool executable entry point')
    parser.add_argument('--dry_run', action='store_true',
        help='print the quantization commands without running them')
    args = parser.parse_args()
    

    compiler_path = args.compiler
    if compiler_path  is None:
        try:
            compiler_path  = Path(os.environ['INTEL_OPENVINO_DIR']) / 'deployment_tools/tools/compile_tool/compile_tool'
        except KeyError:
            sys.exit('Unable to locate compiler tool. '
                + 'Use --compiler or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')
    
    models = common.load_models_from_args(parser, args)
    
    reporter = common.Reporter(common.DirectOutputContext())

    output_dir = args.output_dir or args.model_dir

    for model in models:
        if not model.quantizable:
            reporter.print_section_heading('Skipping {} (quantization not supported)', model.name)
            reporter.print()
            continue

        for precision in sorted(requested_precisions):
            if not quantize(reporter, model, precision, args, output_dir, pot_path, pot_env):
                failed_models.append(model.name)
                break


    if failed_models:
        reporter.print('FAILED:')
        for failed_model_name in failed_models:
            reporter.print(failed_model_name)
        sys.exit(1)

if __name__ == '__main__':
    main()

