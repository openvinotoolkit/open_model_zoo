import argparse
import tensorflow.compat.v1 as tf

from tensorflow.python.tools import optimize_for_inference_lib

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Freeze saved model')

    parser.add_argument('--saved_model_dir', type=str, required=True,
                        help='Path to saved model directory.')
    parser.add_argument('--save_file', type=str, required=True,
                        help='Path to resulting frozen model.')
    return parser.parse_args()

def freeze(saved_model_dir, input_nodes, output_nodes, save_file):
    graph_def = tf.Graph()
    with tf.Session(graph=graph_def) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_nodes
        )
        frozen_graph_def = optimize_for_inference_lib.optimize_for_inference(
            frozen_graph_def,
            input_nodes,
            output_nodes,
            tf.float32.as_datatype_enum
        )
        with open(save_file, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

def main():
    args = parse_args()
    input_nodes = ['map/TensorArrayStack/TensorArrayGatherV3']
    output_nodes = ['softmax_tensor']
    freeze(args.saved_model_dir, input_nodes, output_nodes, args.save_file)

if __name__ == '__main__':
    main()
