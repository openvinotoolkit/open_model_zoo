#!/usr/bin/env python3

# Copyright (c) 2019-2020 Intel Corporation
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
import contextlib
import csv
import itertools
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import timeit
import cv2 as cv
import numpy as np
from pathlib import Path

from args import ArgContext, ModelArg
from cases import DEMOS
from crop_size import CROP_SIZE
from data_sequences import DATA_SEQUENCES
from similarity_measurement import getMSSSIM
from thresholds import THRESHOLDS

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
    parser.add_argument('--devices', default="CPU GPU", # FOR DEBUG: CPU
        help='list of devices to test')
    parser.add_argument('--report-file', type=Path,
        help='path to report file')
    parser.add_argument('--generate-reference', action='store_true',
        help='generate the reference_values.json file')
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
    model_args = [arg
        for demo in demos_to_test
        for case in demo.test_cases
        for arg in case.options.values()
        if isinstance(arg, ModelArg)]

    model_names = {arg.name for arg in model_args}
    model_precisions = {arg.precision for arg in model_args}

    # FOR DEBUG: dl_dir = Path('/home/anthonyquantum') / 'models'
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
        demos_to_test = [demo for demo in DEMOS if demo.full_name in names_of_demos_to_test]
    else:
        demos_to_test = DEMOS

    with temp_dir_as_path() as global_temp_dir:
        dl_dir = prepare_models(auto_tools_dir, args.downloader_cache_dir, args.mo, global_temp_dir, demos_to_test)

        num_failures = 0
        if args.generate_reference:
            reference_values = {}
        else:
            json_file = open(os.path.join(sys.path[0], 'reference_values.json'), 'r')
            reference_values = json.load(json_file)

        os.putenv('PYTHONPATH',  "{}:{}/lib".format(os.environ['PYTHONPATH'], args.demo_build_dir))

        for demo in demos_to_test:
            print('Testing {}...'.format(demo.full_name))
            print()

            declared_model_names = {model['name']
                for model in json.loads(subprocess.check_output(
                    [sys.executable, '--', str(auto_tools_dir / 'info_dumper.py'),
                        '--list', str(demo.models_lst_path(demos_dir))],
                    universal_newlines=True))}

            with temp_dir_as_path() as temp_dir:
                arg_context = ArgContext(
                    source_dir=demos_dir / demo.subdirectory,
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

                    case_model_names = {arg.name for arg in test_case.options.values() if isinstance(arg, ModelArg)}

                    undeclared_case_model_names = case_model_names - declared_model_names
                    if undeclared_case_model_names:
                        print("Test case #{}: models not listed in demo's models.lst: {}".format(
                            test_case_index, ' '.join(sorted(undeclared_case_model_names))))
                        print()

                        num_failures += 1
                        continue

                    for device, dev_arg in device_args.items():
                        print('\nTest case #{}/{}:'.format(test_case_index, device),
                            ' '.join(shlex.quote(str(arg)) for arg in dev_arg + case_args))
                        print(flush=True)
                        try:
                            start_time = timeit.default_timer()
                            subprocess.check_output(fixed_args + dev_arg + case_args,
                                stderr=subprocess.STDOUT, universal_newlines=True)
                            execution_time = timeit.default_timer() - start_time

                            # 'if' is debug-only, won't be needed when all demos will be compatible with ssim test
                            if '-o' in case_args and case_args[case_args.index('-o') + 1] != '.':
                                similarity_res = []

                                out_folder = case_args[case_args.index('-o') + 1]
                                demo_out_file = str(Path(out_folder) / 'out_%08d.BMP')
                                demo_raw_file = str(Path(out_folder) / 'raw_%08d.BMP')
                                out_cap = cv.VideoCapture(demo_out_file)
                                raw_cap = cv.VideoCapture(demo_raw_file)
                                if not out_cap.isOpened() or not raw_cap.isOpened():
                                    raise RuntimeError("Unable to open input files.")
                            
                                while True:
                                    out_ret, out_frame = out_cap.read()
                                    raw_ret, raw_frame = raw_cap.read()

                                    if out_ret and raw_ret:
                                        crop = CROP_SIZE[demo.full_name]
                                        height, width = out_frame.shape[:2]
                                        out_frame = out_frame[crop[0] : height - crop[2], crop[3] : width - crop[1]]
                                        raw_frame = raw_frame[crop[0] : height - crop[2], crop[3] : width - crop[1]]
                                        similarity = list(map(lambda x: round(x, 3),
                                                              getMSSSIM(out_frame, raw_frame)[:-1]))
                                        similarity_res.append(similarity)
                                    else:
                                      break

                                out_cap.release()
                                raw_cap.release()
                                tmp_files = glob.glob(str(Path(out_folder) / '*'))
                                for file_path in tmp_files:
                                    os.remove(file_path)

                                model_name = test_case.options['-m'].name
                                if args.generate_reference:
                                    if demo.full_name not in reference_values:
                                        reference_values[demo.full_name] = {}
                                    reference_values[demo.full_name][model_name] = similarity_res
                                else:
                                    similarity_reference = reference_values[demo.full_name][model_name]
                                    threshold = THRESHOLDS[demo.full_name][model_name]
                                    for i in range(len(similarity_res)):
                                        print("res: {}, ref: {}".format(similarity_res[i], similarity_reference[i]))
                                        # for j in range(len(similarity_res[0])):
                                            # if abs(similarity_res[i][j] - similarity_reference[i][j]) > threshold[j]:
                                                # raise RuntimeError("SSIM test failed: {} != {} with threshold {}."
                                                #                    .format(similarity_res[i], similarity_reference[i],
                                                #                            threshold))
                        except Exception as e:
                            print("Error:")
                            if isinstance(e, subprocess.CalledProcessError):
                                print(e.output)
                                print('Exit code:', e.returncode)
                            else:
                                print(e)

                            num_failures += 1
                            execution_time = -1

                        if args.report_file:
                            collect_result(demo.full_name, device, case_model_names, execution_time, args.report_file)

            print()

    print("Failures: {}".format(num_failures))

    if args.generate_reference:
        with open(os.path.join(sys.path[0], 'reference_values.json'), 'w') as outfile:
            json.dump(reference_values, outfile, indent=2)
            outfile.write("\n")
            print("File reference_values.json was generated.")

    sys.exit(0 if num_failures == 0 else 1)

if __name__ == main():
    main()
