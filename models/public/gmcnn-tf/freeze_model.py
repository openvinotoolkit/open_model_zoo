import argparse
from pathlib import Path
import tensorflow as tf

from net.network import GMCNNModel

def parse_args():
    parser = argparse.ArgumentParser(description='Freeze GMCNN Model')
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to pretrained checkpoints.")
    parser.add_argument("--save_dir", type=Path, required=True,
                        help="Path to write frozen graph.")
    return parser.parse_args()

class Options:
    img_shapes = [512, 680]
    mask_type = 'free_form'
    g_cnum = 32
    d_cnum = 64

def freeze_model(config, input_checkpoints, output_dir):
    model = GMCNNModel()

    with tf.Session() as sess:
        input_image = tf.placeholder(dtype=tf.float32, shape=[None, *config.img_shapes, 3])
        input_mask = tf.placeholder(dtype=tf.float32, shape=[None, *config.img_shapes, 1])

        output = model.evaluate(input_image, input_mask, config)
        output = (output + 1) * 127.5
        output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)

        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = list(map(lambda x: tf.assign(x, tf.train.load_variable(input_checkpoints, x.name)),
                              vars_list))
        sess.run(assign_ops)

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            [output.name.split(':')[0]])

        print("Writing frozen graph...")
        output_file = output_dir / "frozen_model.pb"
        output_file.write_bytes(frozen_graph_def.SerializeToString())

def main():
    args = parse_args()
    config = Options()
    freeze_model(config, args.ckpt_dir, args.save_dir)


if __name__ == "__main__":
    main()
