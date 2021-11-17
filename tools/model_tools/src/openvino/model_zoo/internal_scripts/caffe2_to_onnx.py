import argparse
from pathlib import Path
import sys

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
                        help='Model name to convert.')
    parser.add_argument('--output-file', type=Path, required=True,
                        help='Path to the output ONNX model')
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Path to predict_net .pb file')
    parser.add_argument('--weights', type=Path, required=True,
                        help='Path to init_net .pb file')
    parser.add_argument('--input-shape', metavar='INPUT_DIM', type=positive_int_arg,
                        required=True, help='Shape of the input blob')
    parser.add_argument('--input-names', type=str, required=True,
                        help='Comma separated names of the input layers')

    return parser.parse_args()

def convert_to_onnx(predict_net_path, init_net_path, input_shape, input_names, output_file, model_name=''):
    """Convert Caffe2 model to ONNX and check the resulting onnx model"""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    data_type = onnx.TensorProto.FLOAT
    value_info = {input_names: [data_type, input_shape]}

    predict_net = caffe2_pb2.NetDef()
    predict_net.ParseFromString(predict_net_path.read_bytes())

    predict_net.name = model_name

    init_net = caffe2_pb2.NetDef()
    init_net.ParseFromString(init_net_path.read_bytes())

    onnx_model = Caffe2Frontend.caffe2_net_to_onnx_model(
        predict_net,
        init_net,
        value_info
    )
    try:
        onnx.checker.check_model(onnx_model)
        print('ONNX check passed successfully.')
        output_file.write_bytes(onnx_model.SerializeToString())
    except onnx.onnx_cpp2py_export.checker.ValidationError as exc:
        sys.exit('ONNX check failed with error: ' + str(exc))

def main():
    args = parse_args()
    convert_to_onnx(args.model_path, args.weights, args.input_shape,
        args.input_names, args.output_file, args.model_name
    )


if __name__ == '__main__':
    main()
