# Copyright (c) 2022-2024 Intel Corporation
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

import sys
import argparse
import importlib
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    sys.path.append(str(args.input_dir))
    fsnet = importlib.import_module('frequency_spatial_network')

    under_rate = '20'

    stats = np.load(args.input_dir / "stats_fs_unet_norm_{}.npy".format(under_rate))

    model = fsnet.wnet(stats[0], stats[1], stats[2], stats[3], kshape = (5, 5), kshape2=(3, 3))

    model.load_weights(args.input_dir / "wnet_{}.hdf5".format(under_rate))
    model.save(args.output_dir / 'saved_model')

if __name__ == '__main__':
    main()
