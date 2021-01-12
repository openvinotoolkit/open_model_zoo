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
import importlib
import sys

from pathlib import Path

import tensorflow.compat.v1 as tf

NETWORK_NAME = 'inception_v2'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    sys.path.append(str(args.input_dir / 'models/research/slim'))
    nets_factory = importlib.import_module('nets.nets_factory')

    with tf.Session() as sess:
        network_fn = nets_factory.get_network_fn(NETWORK_NAME, num_classes=1001)
        size = network_fn.default_image_size

        _, end_points = network_fn(tf.placeholder(
            name='input', dtype=tf.float32, shape=(1, size, size, 3)))

        tf.train.Saver().restore(sess, str(args.input_dir / (NETWORK_NAME + '.ckpt')))
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), [end_points['Predictions'].op.name])

        tf.io.write_graph(frozen_graph_def, str(args.output_dir),
            NETWORK_NAME + '.frozen.pb', as_text=False)

if __name__ == '__main__':
    main()
