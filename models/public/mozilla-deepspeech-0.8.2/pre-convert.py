#! /usr/bin/env python3
#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#  This script is automatically executed by open_model_zoo/tools/downloader/converter.py,
#  and runs pbmm_to_pb.py and scorer_to_kenlm.py.
#
import sys
import argparse
import subprocess

from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    subprocess.run([
        sys.executable, '--',
        str(Path(sys.argv[0]).with_name('pbmm_to_pb.py')), '--',
        str(args.input_dir / 'deepspeech-0.8.2-models.pbmm'),
        str(args.output_dir / 'deepspeech-0.8.2-models.pb'),
    ], check=True)

    subprocess.run([
        sys.executable, '--',
        str(Path(sys.argv[0]).with_name('scorer_to_kenlm.py')), '--',
        str(args.input_dir / 'deepspeech-0.8.2-models.scorer'),
        str(args.output_dir / 'deepspeech-0.8.2-models.kenlm'),
    ], check=True)


if __name__ == '__main__':
    main()
