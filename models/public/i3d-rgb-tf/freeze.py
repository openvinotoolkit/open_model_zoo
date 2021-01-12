import tensorflow as tf

import i3d

input = tf.placeholder(tf.float32, shape=(1, 79, 224, 224, 3))

with tf.variable_scope('RGB'):
    model = i3d.InceptionI3d()
    logits, _ = model(input, is_training=False)

variable_map = {}
for variable in tf.global_variables():
    variable_map[variable.name.replace(':0', '')] = variable

saver = tf.train.Saver(var_list=variable_map, reshape=True)

model_logits = logits
model_predictions = tf.nn.softmax(model_logits)

with tf.Session() as sess:
    saver.restore(sess, "data/checkpoints/rgb_imagenet/model.ckpt")
    tf.logging.info('RGB checkpoint restored')

    out_graph = tf.graph_util.convert_variables_to_constants(sess,
        tf.get_default_graph().as_graph_def(), ["Softmax"])

    f = tf.gfile.GFile("i3d-rgb.frozen.pb", "wb")
    f.write(out_graph.SerializeToString())
