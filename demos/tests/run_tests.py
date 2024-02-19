#!/usr/bin/env python3

# Copyright (c) 2019-2024 Intel Corporation
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
* a file how_are_you_doing.wav from (https://storage.openvinotoolkit.org/data/test_data/)
* files stream_1_left.mp4 and stream_1_top.mp4 https://storage.openvinotoolkit.org/data/test_data/videos/smartlab/v3/
"""

import argparse
import certifi
import contextlib
import csv
import json
import os
import shlex
import subprocess  # nosec B404  # disable import-subprocess check
import sys
import tempfile
import timeit
import importlib
import warnings

from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from args import ArgContext, Arg, ModelArg
from cases import Demo
from data_sequences import DATA_SEQUENCES

import requests

scopes = {
    'base': importlib.import_module('cases').DEMOS,
    'performance': importlib.import_module('performance_cases').DEMOS,
}
COCO128_URL = "https://ultralytics.com/assets/coco128.zip"
VIDEO_URL = "https://storage.openvinotoolkit.org/data/test_data/videos/head-pose-face-detection-male.mp4"

def parser_paths_list(supported_devices):
    paths = supported_devices.split(',')
    return [Path(p) for p in paths if Path(p).is_file()]


def build_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument('--demo-build-dir', type=Path, required=True, metavar='DIR',
        help='directory with demo binaries')
    parser.add_argument('--test-data-dir', type=Path, required=True, metavar='DIR',
        help='directory with test data and where to unzip {COCO128_URL}')
    parser.add_argument('--demos', metavar='DEMO[,DEMO...]',
        help='list of demos to run tests for (by default, every demo is tested). '
        'For testing demos of specific implementation pass one (or more) of the next values: cpp, cpp_gapi, python.')
    parser.add_argument('--devices', default="CPU GPU",
        help='list of devices to test')
    parser.add_argument('--downloader-cache-dir', type=Path, metavar='DIR',
        help='directory to use as the cache for the model downloader')
    parser.add_argument('--log-file', type=Path,
        help='path to log file')
    parser.add_argument('--mo', type=Path, metavar='MO.PY',
        help='Model Optimizer entry point script')
    parser.add_argument('--models-dir', type=Path, metavar='DIR',
        help='directory with pre-converted models (IRs)')
    parser.add_argument('--precisions', type=str, nargs='+', default=['FP16', 'FP16-INT8'],
        help='IR precisions for all models. By default, models are tested in FP16, FP16-INT8 precisions')
    parser.add_argument('--report-file', type=Path,
        help='path to report file')
    parser.add_argument('--scope', default='base', choices=('base', 'performance'),
        help='The scenario for testing demos.')
    parser.add_argument('--supported-devices', type=parser_paths_list,
        help='paths to Markdown files with supported devices for each model')
    return parser


def collect_result(demo_name, device, pipeline, execution_time, report_file):
    first_time = not report_file.exists()

    with report_file.open('a+', newline='') as csvfile:
        testwriter = csv.writer(csvfile)
        if first_time:
            testwriter.writerow(["DemoName", "Device", "ModelsInPipeline", "ExecutionTime"])
        testwriter.writerow([demo_name, device, " ".join(sorted(pipeline)), execution_time])


def write_log(test_log, log_file):
    with log_file.open('a+', newline='') as txtfile:
        txtfile.write(test_log + '\n')


@contextlib.contextmanager
def temp_dir_as_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def prepare_models(auto_tools_dir, downloader_cache_dir, mo_path, global_temp_dir, demos_to_test, model_precisions):
    model_names = set()
    model_precisions = set(model_precisions)

    for demo in demos_to_test:
        for case in demo.test_cases:
            for arg in list(case.options.values()) + case.extra_models:
                if isinstance(arg, Arg):
                    for model_request in arg.required_models:
                        model_names.add(model_request.name)

    dl_dir = global_temp_dir / 'models'
    complete_models_lst_path = global_temp_dir / 'models.lst'

    complete_models_lst_path.write_text(''.join(model + '\n' for model in model_names))

    print('Retrieving models...', flush=True)

    try:
        subprocess.check_output(
            [
                sys.executable, '-W', 'error', '--', str(auto_tools_dir / 'downloader.py'),
                '--output_dir', str(dl_dir), '--cache_dir', str(downloader_cache_dir),
                '--list', str(complete_models_lst_path), '--precisions', ','.join(model_precisions),
                '--jobs', '9',
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
                sys.executable, '-W', 'error', '--', str(auto_tools_dir / 'converter.py'),
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


def parse_supported_device_list(paths):
    if not paths:
        return None
    suppressed_devices = {}
    for path in paths:
        with Path(path).open() as f:
            data = f.read()
            rest = '|' + data.split('|', 1)[1]
            table = rest.rsplit('|', 1)[0] + '|'
            result = {}

            for n, line in enumerate(table.split('\n')):
                if n == 0:
                    devices = [t.strip() for t in line.split('|')[2:-1]]
                else:
                    values = [t.strip() for t in line.split('|')[1:-1]]
                    model_name, values = values[0], values[1:]
                    for device, value in zip(devices, values):
                        if not value:
                            result[model_name] = result.get(model_name, []) + [device]
            suppressed_devices.update(result)
    return suppressed_devices


def get_models(case, keys):
    models = []
    for key in keys:
        model = case.options.get(key, None)
        if model:
            models.append(model.name)
    return models


def main():
    warnings.filterwarnings("error")
    args = build_argparser().parse_args()

    DEMOS = scopes[args.scope]
    suppressed_devices = parse_supported_device_list(args.supported_devices)

    omz_dir = (Path(__file__).parent / '../..').resolve()
    demos_dir = omz_dir / 'demos'
    auto_tools_dir = omz_dir / 'tools/model_tools'

    model_info_list = json.loads(subprocess.check_output(
        [sys.executable, '-W', 'error', '--', str(auto_tools_dir / 'info_dumper.py'), '--all'],
        universal_newlines=True))

    model_info = {}
    for model_data in model_info_list:
        models_list = model_data['model_stages'] if model_data['model_stages'] else [model_data]
        for model in models_list:
            model_info[model['name']] = model

    if args.demos is not None:
        names_of_demos_to_test = set(args.demos.split(','))
        if all(impl in Demo.IMPLEMENTATION_TYPES for impl in names_of_demos_to_test):
            names_of_demos_to_test = {demo.subdirectory for demo in DEMOS if demo.implementation in names_of_demos_to_test}

        demos_to_test = [demo for demo in DEMOS if demo.subdirectory in names_of_demos_to_test]
    else:
        demos_to_test = DEMOS

    if len(demos_to_test) == 0:
        if args.demos:
            print("List of demos to test is empty.")
            print(f"Command line argument '--demos {args.demos}' was passed, check that you've specified correct value from the list below:")
            print(*(list(Demo.IMPLEMENTATION_TYPES) + [demo.subdirectory for demo in DEMOS]), sep=',')
        raise RuntimeError("Not found demos to test!")

    print(f"{len(demos_to_test)} demos will be tested:")
    print(*[demo.subdirectory for demo in demos_to_test], sep =',')
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()
    with urlopen(COCO128_URL) as zipresp:  # nosec B310  # disable urllib_urlopen because url is hardcoded
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(args.test_data_dir)

    r = requests.get(VIDEO_URL)
    with open("video.mp4", 'wb') as f:
        f.write(r.content)

    with temp_dir_as_path() as global_temp_dir:
        if args.models_dir:
            dl_dir = args.models_dir
            print(f"\nRunning on pre-converted IRs: {str(dl_dir)}\n")
        else:
            dl_dir = prepare_models(auto_tools_dir, args.downloader_cache_dir, args.mo, global_temp_dir, demos_to_test, args.precisions)

        num_failures = 0

        try:
            pythonpath = f"{os.environ['PYTHONPATH']}{os.pathsep}"
        except KeyError:
            pythonpath = ''
        demo_environment = {**os.environ,
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONPATH': f"{pythonpath}{args.demo_build_dir}",
        }

        print('Demo Environment: {}'.format(demo_environment))

        failed_tests = []
        for demo in demos_to_test:
            header = 'Testing {}'.format(demo.subdirectory)
            print(header)
            print()
            demo.set_precisions(args.precisions, model_info)

            declared_model_names = set()
            for model_data in json.loads(subprocess.check_output(
                    [sys.executable, '-W', 'error', '--', str(auto_tools_dir / 'info_dumper.py'),
                        '--list', str(demo.models_lst_path(demos_dir))],
                    universal_newlines=True)):
                models_list = model_data['model_stages'] if model_data['model_stages'] else [model_data]
                for model in models_list:
                    declared_model_names.add(model['name'])

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
                    test_case_models = get_models(test_case, demo.model_keys)

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
                        skip = False
                        for model in test_case_models:
                            if suppressed_devices and device in suppressed_devices.get(model, []):
                                print('Test case #{}/{}: Model {} is suppressed on device'
                                      .format(test_case_index, device, model))
                                print(flush=True)
                                skip = True
                        if skip: continue
                        test_descr = 'Test case #{}/{}:\n{}'.format(test_case_index, device, ' '.join(shlex.quote(str(arg))
                            for arg in fixed_args + dev_arg + case_args))
                        print(test_descr)
                        print(flush=True)
                        try:
                            start_time = timeit.default_timer()
                            output = subprocess.check_output(fixed_args + dev_arg + case_args,
                                stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8',
                                env=demo_environment, timeout=600)
                            execution_time = timeit.default_timer() - start_time
                            demo.parse_output(output, test_case, device)
                        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                            output = e.output
                            if isinstance(e, subprocess.CalledProcessError):
                                exit_msg = f'Exit code: {e.returncode}\n'
                            elif isinstance(e, subprocess.TimeoutExpired):
                                exit_msg = f'Command timed out after {e.timeout} seconds\n'
                            output += exit_msg
                            print(output)
                            failed_tests.append(test_descr + '\n' + exit_msg)
                            num_failures += 1
                            execution_time = -1

                        if args.report_file:
                            collect_result(demo.subdirectory, device, case_model_names, execution_time, args.report_file)
                        if args.log_file:
                            if test_case_index == 0:
                                write_log(header, args.log_file)
                            write_log(test_descr, args.log_file)
                            write_log(output, args.log_file)

            print()

    print("{} failures:".format(num_failures))
    for test in failed_tests:
        print(test)

    sys.exit(0 if num_failures == 0 else 1)


if __name__ == '__main__':
    main()
