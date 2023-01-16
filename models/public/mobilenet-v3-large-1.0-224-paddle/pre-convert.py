#!/usr/bin/env python3

# Copyright (c) 2022-2023 Intel Corporation
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
import subprocess # nosec - disable B404:import-subprocess check
import sys

from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    subprocess.run([sys.executable, '--',
        str(args.input_dir / 'tools' / 'export_model.py'), '-c',
        str(args.input_dir / 'MobileNetV3_large_x1_0.yaml'), '-o',
        f"Global.pretrained_model={args.input_dir/ 'MobileNetV3_large_x1_0_pretrained'}",
        '-o', f"Global.save_inference_dir={args.output_dir / 'inference_model'}",
        '-o', 'Global.device=cpu'
    ], check=True)


if __name__ == '__main__':
    main()
