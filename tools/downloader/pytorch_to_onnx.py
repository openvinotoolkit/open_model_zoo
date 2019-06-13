import argparse
import sys

import torch.onnx


TORCHVISION_MODELS = [
    'resnet-v1-50',
    'inception-v3'
]

PUBLIC_PYTORCH_MODELS = [
    'mobilenet-v2'
]

SUPPORTED_MODELS = TORCHVISION_MODELS + PUBLIC_PYTORCH_MODELS


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

    parser = argparse.ArgumentParser(description='Conversion script for pretrained models from PyTorch to ONNX')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model name to convert.'
                             ' One of the ' + ', '.join(SUPPORTED_MODELS) + ' is supported')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the PyTorch pretrained weights')
    parser.add_argument('--input-shape', metavar='INPUT_DIM', type=positive_int_arg, required=True,
                        help='Shape of the input blob')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to the output ONNX model')

    parser.add_argument('--model-path', type=str,
                        help='Path to python model description if model is not a standard torchvision model')
    parser.add_argument('--input-names', type=str, nargs='+',
                        help='Space separated names of the input layers')
    parser.add_argument('--output-names', type=str, nargs='+',
                        help='Space separated names of the output layers')
    return parser.parse_args()


def load_model(model_name, weights, model_path=None):
    """Import model and load pretrained weights"""

    if model_name not in SUPPORTED_MODELS:
        sys.exit('Only ' + ', '.join(SUPPORTED_MODELS) + ' are available for conversion')

    if model_name in TORCHVISION_MODELS:
        try:
            import torchvision
            if model_name == 'resnet-v1-50':
                model = torchvision.models.resnet50()
            elif model_name == 'inception-v3':
                model = torchvision.models.inception_v3()
        except ImportError as exc:
            print(exc)
            sys.exit('The torchvision package was not found.'
                     'Please install it to default location or '
                     'update PYTHONPATH environment variable '
                     'with the path to the installed torchvision package.')
    else:
        sys.path.append(model_path)
        try:
            from MobileNetV2 import MobileNetV2
            model = MobileNetV2(n_class=1000)
        except ImportError as exc:
            print(exc)
            sys.exit('MobileNetV2 can not be imported. ' +
                     'Please provide valid path to python model description with use of --model-path argument')

    model.load_state_dict(torch.load(weights, map_location='cpu'))
    return model


def convert_to_onnx(pytorch_model, input_shape, output_file, input_names, output_names):
    """Convert PyTorch model to ONNX and check the resulting onnx model"""

    pytorch_model.eval()
    onnx_input_shape = torch.randn(input_shape)
    torch.onnx.export(pytorch_model, onnx_input_shape, output_file,
                      verbose=False, input_names=input_names, output_names=output_names)

    # Model check after conversion
    import onnx
    model_from_onnx = onnx.load(output_file)
    try:
        onnx.checker.check_model(model_from_onnx)
        print('ONNX check passed successfully.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as exc:
        sys.exit('ONNX check failed with error: ' + str(exc))


def main():
    args = parse_args()
    model = load_model(args.model_name, args.weights, args.model_path)
    convert_to_onnx(model, args.input_shape, args.output_file, args.input_names, args.output_names)


if __name__ == '__main__':
    main()
