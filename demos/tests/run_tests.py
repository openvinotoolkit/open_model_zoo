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

For the tests to work, the test data directory must contain:
* a "BraTS" subdirectory with brain tumor dataset in NIFTI format (see http://medicaldecathlon.com,
  https://drive.google.com/open?id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU);
* a "ILSVRC2012_img_val" subdirectory with the ILSVRC2012 dataset;
* a "Image_Retrieval" subdirectory with image retrieval dataset (images, videos) (see https://github.com/19900531/test)
  and list of images (see https://github.com/opencv/openvino_training_extensions/blob/develop/tensorflow_toolkit/image_retrieval/data/gallery/gallery.txt)
"""

import argparse
import collections
import csv
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import timeit

from pathlib import Path

from args import ArgContext, ModelArg
from cases import DEMOS
from data_sequences import DATA_SEQUENCES

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
    parser.add_argument('--mo', type=Path, metavar='MO.PY',
        help='Model Optimizer entry point script')
    parser.add_argument('--devices', default="CPU GPU",
        help='list of devices to test')
    parser.add_argument('--report-file', type=Path,
        help='path to report file')
    return parser.parse_args()

def collect_result(demo_name, device, pipeline, execution_time, report_file):
    first_time = not report_file.exists()
    pipeline.sort()
    with report_file.open('a+', newline='') as csvfile:
        testwriter = csv.writer(csvfile)
        if first_time:
            testwriter.writerow(["DemoName", "Device", "ModelsInPipeline", "ExecutionTime"])
        testwriter.writerow([demo_name, device, " ".join(pipeline), execution_time])

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

    os.putenv('PYTHONPATH',  "{}:{}/lib".format(os.environ['PYTHONPATH'], args.demo_build_dir))

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

            try:
                subprocess.check_output(
                    [
                        sys.executable, '--', str(auto_tools_dir / 'converter.py'),
                        '--download_dir', str(dl_dir), '--list', str(demo.models_lst_path(demos_dir)), '--jobs', 'auto',
                    ] + ([] if args.mo is None else ['--mo', str(args.mo)]),
                    stderr=subprocess.STDOUT, universal_newlines=True)
            except subprocess.CalledProcessError as e:
                print(e.output)
                print('Exit code:', e.returncode)
                num_failures += len(demo.test_cases)
                continue

            print()

            arg_context = ArgContext(
                source_dir=demos_dir / demo.subdirectory,
                dl_dir=dl_dir,
                data_sequence_dir=Path(temp_dir) / 'data_seq',
                data_sequences=DATA_SEQUENCES,
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
            device_args = demo.device_args(args.devices.split())
            for test_case_index, test_case in enumerate(demo.test_cases):

                case_args = [demo_arg
                    for key, value in sorted(test_case.options.items())
                    for demo_arg in option_to_args(key, value)]

                pipeline = [value.name for key, value in test_case.options.items() if isinstance(value, ModelArg)]

                for device, dev_arg in device_args.items():
                    print('Test case #{}/{}:'.format(test_case_index, device),
                        ' '.join(shlex.quote(str(arg)) for arg in dev_arg + case_args))
                    print(flush=True)
                    try:
                        start_time = timeit.default_timer()
                        subprocess.check_output(fixed_args + dev_arg + case_args,
                            stderr=subprocess.STDOUT, universal_newlines=True)
                        execution_time = timeit.default_timer() - start_time
                    except subprocess.CalledProcessError as e:
                        print(e.output)
                        print('Exit code:', e.returncode)
                        num_failures += 1
                        execution_time = -1

                    if args.report_file:
                        collect_result(demo.full_name, device, pipeline, execution_time, args.report_file)

        print()

    print("Failures: {}".format(num_failures))

    sys.exit(0 if num_failures == 0 else 1)

if __name__ == main():
    main()
