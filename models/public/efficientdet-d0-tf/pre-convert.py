#!/usr/bin/env python3

# Copyright (c) 2020 Intel Corporation
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
import shutil
import subprocess
import sys

from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    saved_model_dir = args.output_dir / "efficientdet-d0_saved_model"
    if saved_model_dir.exists():
        shutil.rmtree(str(saved_model_dir))

    subprocess.run([sys.executable, '--',
        str(args.input_dir / 'model/model_inspect.py'),
        "--runmode=saved_model",
        "--model_name=efficientdet-d0",
        "--ckpt_path={}".format(args.input_dir / "efficientdet-d0"),
        "--saved_model_dir={}".format(saved_model_dir),
    ], check=True)

if __name__ == '__main__':
    main()
