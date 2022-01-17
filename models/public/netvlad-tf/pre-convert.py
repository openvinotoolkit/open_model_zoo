#!/usr/bin/env python3

# Copyright (c) 2022 Intel Corporation
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

NETWORK_NAME = 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    tf.disable_eager_execution()

    sys.path.append(str(args.input_dir))
    nets = importlib.import_module('netvlad_tf.nets')

    image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    net_out = nets.vgg16NetvladPca(image_batch)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, str(args.input_dir / NETWORK_NAME / NETWORK_NAME))
        graph_def_frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(),
                                                                        [net_out.op.name])

    tf.io.write_graph(graph_def_frozen, str(args.output_dir), 'model_frozen.pb', as_text=False)


if __name__ == '__main__':
    main()
