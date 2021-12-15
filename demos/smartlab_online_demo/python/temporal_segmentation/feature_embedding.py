# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from temporal_segmentation import preprocessing
import os
import i3d
from config_online import online_opt as opt


def embed_model_initialization(batch_size, embed_frame_length):
    """
    预先进行embedding model的初始化
    Args:
        batch_size: embed_batch_size
        embed_frame_length: 采样帧长

    Returns: embedding model and the sess
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.logging.set_verbosity(tf.logging.INFO)

    eval_type = 'rgb'
    imagenet_pretrained = True
    rgb_input = tf.placeholder(tf.float32,
                               shape=(batch_size, embed_frame_length, opt.img_size_h, opt.img_size_w, 3))

    with tf.variable_scope('RGB'):
        rgb_model = i3d.InceptionI3d(num_classes=400, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            if eval_type == 'rgb600':
                rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
            else:
                rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    model_logits = rgb_logits
    model_predictions = tf.nn.softmax(model_logits)

    sess = tf.Session()
    feed_dict = {}
    if eval_type in ['rgb', 'rgb600', 'joint']:
        if imagenet_pretrained:
            rgb_saver.restore(sess, opt.embed_model_dir['rgb_imagenet'])
        else:
            rgb_saver.restore(sess, opt.embed_model_dir[eval_type])
    tf.logging.info('RGB checkpoint restored')

    return rgb_model, sess


def embed_model_inference(embed_sess, embed_model, input_data):
    """
    I3D模型在线推理
    Args:
        embed_sess: initialized session
        embed_model:  initialized embedding model
        input_data: BxNxHxWx3

    Returns:
        out_logits:BxC(embed_dim)
    """
    rgb_input = tf.placeholder(tf.float32, shape=(opt.embed_batch_size, opt.embed_window_length,
                                                  opt.img_size_h, opt.img_size_w, 3))
    with tf.variable_scope('RGB'):
        rgb_model = embed_model
        rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

    model_logits = rgb_logits
    model_predictions = tf.nn.softmax(model_logits)

    feed_dict = {}
    feed_dict[rgb_input] = input_data
    out_logits, _ = embed_sess.run([model_logits, model_predictions], feed_dict=feed_dict)
    out_logits = np.array(out_logits).T
    return out_logits  # C x B


if __name__ == '__main__':
    embed_batch_size = opt.embed_batch_size
    embed_frame_length = opt.embed_frame_length
    img_size_h = opt.img_size_h
    img_size_w = opt.img_size_w
    total_frame = 3462
    data_path = "data/tianping/raw_data/jeffrey_wrong_top_1_2"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.logging.set_verbosity(tf.logging.INFO)

    # ======================== Embedding Model Initialization ========================
    embed_model, embed_sess = embed_model_initialization(batch_size=embed_batch_size, embed_frame_length=embed_frame_length)

    # ======================== Embedding Inference ========================
    start_frame = 0
    num_batch = (total_frame - embed_frame_length) // embed_batch_size
    print("num_batch:", num_batch)
    features = np.zeros((opt.embed_dim, embed_batch_size * num_batch))
    for i in range(num_batch):
        print("batch_i:", i)
        input_data = preprocessing.produce_batch(data_path, embed_frame_length, start_frame + i * embed_batch_size,
                                                 opt.img_size_h, opt.img_size_w, embed_batch_size)
        embed_features = embed_model_inference(embed_sess=embed_sess, embed_model=embed_model, input_data=input_data)

        features[:, i * embed_batch_size:(i + 1) * embed_batch_size] = embed_features

    embed_sess.close()
