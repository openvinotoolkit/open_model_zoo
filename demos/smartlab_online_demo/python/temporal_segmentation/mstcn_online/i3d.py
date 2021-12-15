# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.

The model is introduced in:

  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
  Joao Carreira, Andrew Zisserman
  https://arxiv.org/pdf/1705.07750v1.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf


class Unit3D(snt.AbstractModule):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(self, output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation_fn=tf.nn.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def _build(self, inputs, is_training):
        """Connects the module to inputs.

        Args:
          inputs: Inputs to the Unit3D component.
          is_training: whether to use training mode for snt.BatchNorm (boolean).

        Returns:
          Outputs from the module.
        """
        net = snt.Conv3D(output_channels=self._output_channels,
                         kernel_shape=self._kernel_shape,
                         stride=self._stride,
                         padding=snt.SAME,
                         use_bias=self._use_bias)(inputs)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net, is_training=is_training, test_local_stats=False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net


class InceptionI3d(snt.AbstractModule):
    """Inception-v1 I3D architecture.

    The model is introduced in:

      Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
      Joao Carreira, Andrew Zisserman
      https://arxiv.org/pdf/1705.07750v1.pdf.

    See also the Inception architecture, introduced in:

      Going deeper with convolutions
      Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
      Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
      http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d'):
        """Initializes I3D model instance.

        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.

        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__(name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs, is_training, dropout_keep_prob=1.0):
        """Connects the model to inputs.

        Args:
          inputs: Inputs to the model, which should have dimensions
              `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
          is_training: whether to use training mode for snt.BatchNorm (boolean).
          dropout_keep_prob: Probability for the tf.nn.dropout layer (float in
              [0, 1)).

        Returns:
          A tuple consisting of:
            1. Network output at location `self._final_endpoint`.
            2. Dictionary containing all endpoints up to `self._final_endpoint`,
               indexed by endpoint name.

        Raises:
          ValueError: if `self._final_endpoint` is not recognized.
        """
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        net = inputs
        end_points = {}
        end_point = 'Conv3d_1a_7x7'
        net = Unit3D(output_channels=64, kernel_shape=[7, 7, 7],
                     stride=[2, 2, 2], name=end_point)(net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool3d_2a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        end_point = 'Conv3d_2b_1x1'
        net = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                     name=end_point)(net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        end_point = 'Conv3d_2c_3x3'
        net = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                     name=end_point)(net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool3d_3a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=32, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_4a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=208, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=48, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=224, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=256, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=144, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=288, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_5a_2x2'
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0a_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_1,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_2'):
                branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                  name='Conv3d_0b_3x3')(branch_2,
                                                        is_training=is_training)
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
                                   strides=[1, 1, 1, 1, 1], padding=snt.VALID)
            # net = tf.nn.dropout(net, dropout_keep_prob)
            net = tf.nn.dropout(net, rate=1 - dropout_keep_prob)
            if self._spatial_squeeze:
                nets = tf.squeeze(net, [2, 3], name='SpatialSqueeze')
        avg_net = tf.reduce_mean(nets, axis=1)
        if self._final_endpoint == end_point: return avg_net, end_points
