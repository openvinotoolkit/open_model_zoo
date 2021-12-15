# -*- coding: utf-8 -*-
import time
import os
import numpy as np
import tensorflow as tf
import i3d


def embed_model_initialize(opt):
    """
        预先进行embedding model的初始化
    Returns: embedding model and the sess
    """
    batch_size = opt.embed_batch_size
    embed_frame_length = opt.embed_window_length

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    eval_type = 'rgb'
    imagenet_pretrained = True
    rgb_input = tf.placeholder(tf.float32, shape=(batch_size, embed_frame_length, opt.img_size_h, opt.img_size_w, 3))

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

    sess = tf.Session(config=config)
    if eval_type in ['rgb', 'rgb600', 'joint']:
        if imagenet_pretrained:
            rgb_saver.restore(sess, str(opt.embed_model_dir['rgb_imagenet']))
        else:
            rgb_saver.restore(sess, str(opt.embed_model_dir[eval_type]))
    tf.logging.info('RGB checkpoint restored')

    return rgb_model, sess


def embed_model_inference(embed_sess, embed_model, input_data, opt):
    """
    I3D模型在线推理
    Args:
        embed_sess: initialized session
        embed_model:  initialized embedding model
        input_data: BxNxHxWx3

    Returns:
        out_logits:BxC(embed_dim)
    """
    time1 = time.time()
    rgb_input = tf.placeholder(tf.float32, shape=(opt.embed_batch_size, opt.embed_window_length,
                                                  opt.img_size_h, opt.img_size_w, 3))
    with tf.variable_scope('RGB'):
        rgb_model = embed_model
        rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

    model_logits = rgb_logits
    model_predictions = tf.nn.softmax(model_logits)
    time2 = time.time()
    print("Initialize time:", time2 - time1)
    feed_dict = {}
    feed_dict[rgb_input] = input_data
    out_logits, _ = embed_sess.run([model_logits, model_predictions], feed_dict=feed_dict)
    out_logits = np.array(out_logits).T
    return out_logits  # C x B
