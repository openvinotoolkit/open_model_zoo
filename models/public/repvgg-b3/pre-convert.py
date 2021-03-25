#!/usr/bin/env python3

# Copyright (c) 2021 Intel Corporation
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
import importlib
from torch import load
import sys

from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    sys.path.append(str(args.input_dir))
    repvgg = importlib.import_module('repvgg')

    train_model = repvgg.create_RepVGG_B3(deploy=False)

    checkpoint = load(str(args.input_dir / 'RepVGG-B3-200epochs-train.pth'))
    ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    train_model.load_state_dict(ckpt)

    repvgg.repvgg_model_convert(train_model, build_func=repvgg.create_RepVGG_B3,
                                save_path=str(args.output_dir / 'RepVGG-B3-200epochs.pth'))


if __name__ == '__main__':
    main()
