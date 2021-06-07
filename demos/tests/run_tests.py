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
  and list of images (see https://github.com/openvinotoolkit/training_extensions/blob/089de2f/misc/tensorflow_toolkit/image_retrieval/data/gallery/gallery.txt)
* a "msasl" subdirectory with the MS-ASL dataset (https://www.microsoft.com/en-us/research/project/ms-asl/)
* a file how_are_you_doing.wav from <openvino_dir>/deployment_tools/demo/how_are_you_doing.wav
"""

import argparse
import contextlib
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import timeit

from pathlib import Path

from args import ArgContext, Arg, ModelArg
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

    with report_file.open('a+', newline='') as csvfile:
        testwriter = csv.writer(csvfile)
        if first_time:
            testwriter.writerow(["DemoName", "Device", "ModelsInPipeline", "ExecutionTime"])
        testwriter.writerow([demo_name, device, " ".join(sorted(pipeline)), execution_time])


@contextlib.contextmanager
def temp_dir_as_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def prepare_models(auto_tools_dir, downloader_cache_dir, mo_path, global_temp_dir, demos_to_test):
    model_names = set()
    model_precisions = set()

    for demo in demos_to_test:
        for case in demo.test_cases:
            for arg in list(case.options.values()) + case.extra_models:
                if isinstance(arg, Arg):
                    for model_request in arg.required_models:
                        model_names.add(model_request.name)
                        model_precisions.update(model_request.precisions)

    if not model_precisions:
        model_precisions.add('FP32')

    dl_dir = global_temp_dir / 'models'
    complete_models_lst_path = global_temp_dir / 'models.lst'

    complete_models_lst_path.write_text(''.join(model + '\n' for model in model_names))

    print('Retrieving models...', flush=True)

    try:
        subprocess.check_output(
            [
                sys.executable, '--', str(auto_tools_dir / 'downloader.py'),
                '--output_dir', str(dl_dir), '--cache_dir', str(downloader_cache_dir),
                '--list', str(complete_models_lst_path), '--precisions', ','.join(model_precisions),
            ],
            stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        print('Exit code:', e.returncode)
        sys.exit(1)

    print()
    print('Converting models...', flush=True)

    try:
        subprocess.check_output(
            [
                sys.executable, '--', str(auto_tools_dir / 'converter.py'),
                '--download_dir', str(dl_dir), '--list', str(complete_models_lst_path),
                '--precisions', ','.join(model_precisions), '--jobs', 'auto',
                *(['--mo', str(mo_path)] if mo_path else []),
            ],
            stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        print('Exit code:', e.returncode)
        sys.exit(1)

    print()

    return dl_dir


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
        names_of_demos_to_test = set(args.demos.split(','))
        demos_to_test = [demo for demo in DEMOS if demo.subdirectory in names_of_demos_to_test]
    else:
        demos_to_test = DEMOS

    with temp_dir_as_path() as global_temp_dir:
        dl_dir = prepare_models(auto_tools_dir, args.downloader_cache_dir, args.mo, global_temp_dir, demos_to_test)

        num_failures = 0

        python_module_subdir = "" if platform.system() == "Windows" else "/lib"
        demo_environment = {**os.environ,
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONPATH': f"{os.environ['PYTHONPATH']}{os.pathsep}{args.demo_build_dir}{python_module_subdir}",
        }

        for demo in demos_to_test:
            print('Testing {}...'.format(demo.subdirectory))
            print()

            declared_model_names = {model['name']
                for model in json.loads(subprocess.check_output(
                    [sys.executable, '--', str(auto_tools_dir / 'info_dumper.py'),
                        '--list', str(demo.models_lst_path(demos_dir))],
                    universal_newlines=True))}

            with temp_dir_as_path() as temp_dir:
                arg_context = ArgContext(
                    dl_dir=dl_dir,
                    data_sequence_dir=temp_dir / 'data_seq',
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

                    case_model_names = {arg.name for arg in list(test_case.options.values()) + test_case.extra_models if isinstance(arg, ModelArg)}

                    undeclared_case_model_names = case_model_names - declared_model_names
                    if undeclared_case_model_names:
                        print("Test case #{}: models not listed in demo's models.lst: {}".format(
                            test_case_index, ' '.join(sorted(undeclared_case_model_names))))
                        print()

                        num_failures += 1
                        continue

                    for device, dev_arg in device_args.items():
                        print('Test case #{}/{}:'.format(test_case_index, device),
                            ' '.join(shlex.quote(str(arg)) for arg in dev_arg + case_args))
                        print(flush=True)
                        try:
                            start_time = timeit.default_timer()
                            subprocess.check_output(fixed_args + dev_arg + case_args,
                                stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8',
                                env=demo_environment)
                            execution_time = timeit.default_timer() - start_time
                        except subprocess.CalledProcessError as e:
                            print(e.output)
                            print('Exit code:', e.returncode)
                            num_failures += 1
                            execution_time = -1

                        if args.report_file:
                            collect_result(demo.subdirectory, device, case_model_names, execution_time, args.report_file)

            print()

    print("Failures: {}".format(num_failures))

    sys.exit(0 if num_failures == 0 else 1)


if __name__ == '__main__':
    main()
