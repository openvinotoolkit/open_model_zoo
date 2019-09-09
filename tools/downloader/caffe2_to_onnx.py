import argparse
from pathlib import Path
import sys
import json
import os

import onnx
from caffe2.python.onnx.frontend import Caffe2Frontend
from caffe2.proto import caffe2_pb2


def positive_int_arg(values):
    """Check positive integer type for input argument"""
    result = []
    for value in values.split(','):
        try:
            ivalue = int(value)
            if ivalue < 0:
                raise argparse.ArgumentTypeError('Argument must be a positive integer')
            result.append(ivalue)
        except Exception as exc:
            print(exc)
            sys.exit('Invalid value for input argument: {!r}, a positive integer is expected'.format(value))
    return result

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Conversion of pretrained models from Caffe2 to ONNX')

    parser.add_argument('--model-name', type=str, required=True,
                        help='Model to convert. May be class name or name of constructor function')
    parser.add_argument('--output-file', type=Path, required=True,
                        help='Path to the output ONNX model')
    parser.add_argument('--predict-net-path', type=str, required=True,
                        help='Path to predict_net .pb file')
    parser.add_argument('--init-net-path', type=str, required=True,
                        help='Path to init_net .pb file')
    parser.add_argument('--input-shape', metavar='INPUT_DIM', type=positive_int_arg, nargs='+',
                        required=True, help='Shape of the input blob')
    parser.add_argument('--input-names', type=str, nargs='+',
                        help='Space separated names of the input layers')

    return parser.parse_args()

def load_model(predict_net_path, init_net_path):
    predict_net = caffe2_pb2.NetDef()
    with open(predict_net_path, 'rb') as file:
        predict_net.ParseFromString(file.read())

    init_net = caffe2_pb2.NetDef()
    with open(init_net_path, 'rb') as file:
        init_net.ParseFromString(file.read())

    return predict_net, init_net

def convert_to_onnx(predict_net, init_net, input_shape, input_names, output_file, model_name=''):
    """Convert Caffe2 model to ONNX and check the resulting onnx model"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    value_info = {}
    for name, shape in zip(input_names, input_shape):
        value_info[name] = [shape[0], shape]
    if predict_net.name == "":
        predict_net.name = model_name

    onnx_model = Caffe2Frontend.caffe2_net_to_onnx_model(
        predict_net,
        init_net,
        value_info,
    )
    try:
        onnx.checker.check_model(onnx_model)
        print('ONNX check passed successfully.')
        with open(str(output_file), 'wb') as f:
            f.write(onnx_model.SerializeToString())
    except onnx.onnx_cpp2py_export.checker.ValidationError as exc:
        sys.exit('ONNX check failed with error: ' + str(exc))

def main():
    args = parse_args()
    predict_net, init_net = load_model(args.predict_net_path, args.init_net_path)
    convert_to_onnx(predict_net, init_net, args.input_shape, args.input_names, args.output_file, args.model_name)


if __name__ == '__main__':
    main()