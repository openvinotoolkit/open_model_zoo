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

"""
Test script for the demos.

For the tests to work, the test data directory must contain a "ILSVRC2012_img_val"
subdirectory with the ILSVRC2012 dataset.
"""

import argparse
import collections
import itertools
import json
import shlex
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path

from args import ArgContext
from cases import DEMOS
from image_sequences import IMAGE_SEQUENCES

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument('--demo-build-dir', type=Path, required=True, metavar='DIR',
        help='directory with demo binaries')
    parser.add_argument('--test-data-dir', type=Path, required=True, metavar='DIR',
        help='directory with test data')
    parser.add_argument('--downloader-cache-dir', type=Path, required=True, metavar='DIR',
        help='directory to use as the cache for the model downloader')
    parser.add_argument('--demos', metavar='DEMO[,DEMO...]',
        help='list of demos to run tests for (by default, every demo is tested)')
    return parser.parse_args()

def main():
    args = parse_args()

    omz_dir = (Path(__file__).parent / '../..').resolve()
    demos_dir = omz_dir / 'demos'
    auto_tools_dir = omz_dir / 'tools/downloader'

    model_info_list = json.loads(subprocess.check_output(
        [sys.executable, '--', str(auto_tools_dir / 'info_dumper.py'), '--all'],
        universal_newlines=True))
    model_info = {model['name']: model for model in model_info_list}

    if args.demos is not None:
        demos_to_test = set(args.demos.split(','))
    else:
        demos_to_test = {demo.full_name for demo in DEMOS}

    num_failures = 0

    for demo in DEMOS:
        if demo.full_name not in demos_to_test: continue

        print('Testing {}...'.format(demo.full_name))
        print()

        with tempfile.TemporaryDirectory() as temp_dir:
            dl_dir = Path(temp_dir) / 'models'

            print('Retrieving models...', flush=True)

            try:
                subprocess.check_output(
                    [
                        sys.executable, '--', str(auto_tools_dir / 'downloader.py'),
                        '--output_dir', str(dl_dir), '--cache_dir', str(args.downloader_cache_dir),
                        '--list', str(demo.models_lst_path(demos_dir)),
                    ],
                    stderr=subprocess.STDOUT, universal_newlines=True)
            except subprocess.CalledProcessError as e:
                print(e.output)
                print('Exit code:', e.returncode)
                num_failures += len(demo.test_cases)
                continue

            print()

            arg_context = ArgContext(
                dl_dir=dl_dir,
                image_sequence_dir=Path(temp_dir) / 'image_seq',
                image_sequences=IMAGE_SEQUENCES,
                model_info=model_info,
                test_data_dir=args.test_data_dir,
            )

            def resolve_arg(arg):
                if isinstance(arg, str): return arg
                return arg.resolve(arg_context)

            def option_to_args(key, value):
                if value is None: return [key]
                if isinstance(value, list): return [key, *map(resolve_arg, value)]
                return [key, resolve_arg(value)]

            fixed_args = demo.fixed_args(demos_dir, args.demo_build_dir)

            print('Fixed arguments:', ' '.join(map(shlex.quote, fixed_args)))
            print()

            for test_case_index, test_case in enumerate(demo.test_cases):
                case_args = [demo_arg
                    for key, value in sorted(test_case.options.items())
                    for demo_arg in option_to_args(key, value)]

                print('Test case #{}:'.format(test_case_index + 1),
                    ' '.join(shlex.quote(str(arg)) for arg in case_args))
                print(flush=True)

                try:
                    subprocess.check_output(fixed_args + case_args,
                        stderr=subprocess.STDOUT, universal_newlines=True)
                except subprocess.CalledProcessError as e:
                    print(e.output)
                    print('Exit code:', e.returncode)
                    num_failures += 1

        print()

    print("Failures: {}".format(num_failures))

    sys.exit(0 if num_failures == 0 else 1)

if __name__ == main():
    main()
