"""
Copyright (C) 2023 KNS Group LLC (YADRO)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import sys
import subprocess  # nosec - disable B404:import-subprocess check

from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    saved_model_dir = args.output_dir

    subprocess.run([sys.executable, '--',
                    str(args.input_dir / 'convert_model.py'),
                    "--features", "hdr", "alb",
                    "--input_names", "color", "albedo",
                    f'--input_path_tza={args.input_dir / "denoise_rt_hdr_alb.tza"}',
                    f'--output_path_onnx={saved_model_dir / "denoise_rt_hdr_alb.onnx"}'
                    ], check=True)


if __name__ == '__main__':
    main()
