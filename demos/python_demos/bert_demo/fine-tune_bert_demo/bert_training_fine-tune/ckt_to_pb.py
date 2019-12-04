# -*- coding: utf-8 -*-
import tensorflow as tf


def freeze_graph(input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: 
    :return:
    '''
 
    
    output_node_names = "bert/pooler/dense/Tanh"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) 
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))
 
        with tf.gfile.GFile(output_graph, "wb") as f: 
            f.write(output_graph_def.SerializeToString()) 
        print("%d ops in the final graph." % len(output_graph_def.node))


input_checkpoint='result2/model.ckpt-468'
out_pb_path="bert-finetune.pb"
freeze_graph(input_checkpoint,out_pb_path)
