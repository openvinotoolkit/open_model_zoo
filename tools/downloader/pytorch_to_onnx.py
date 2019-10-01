import argparse
from pathlib import Path
import sys

import onnx
import torch
import torch.onnx


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

    parser = argparse.ArgumentParser(description='Conversion of pretrained models from PyTorch to ONNX')

    parser.add_argument('--model-name', type=str, required=True,
                        help='Model to convert. May be class name or name of constructor function')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the weights in PyTorch\'s format')
    parser.add_argument('--input-shape', metavar='INPUT_DIM', type=positive_int_arg, required=True,
                        help='Shape of the input blob')
    parser.add_argument('--output-file', type=Path, required=True,
                        help='Path to the output ONNX model')
    parser.add_argument('--from-torchvision', action='store_true',
                        help='Sets model\'s origin as Torchvision*')
    parser.add_argument('--model-path', type=str,
                        help='Path to PyTorch model\'s source code if model is not from Torchvision*')
    parser.add_argument('--import-module', type=str, default='',
                        help='Name of module, which contains model\'s constructor.'
                        'Requires if model not from Torchvision')
    parser.add_argument('--input-names', type=str, nargs='+',
                        help='Space separated names of the input layers')
    parser.add_argument('--output-names', type=str, nargs='+',
                        help='Space separated names of the output layers')

    return parser.parse_args()


def load_model(model_name, weights, from_torchvision=True, model_path=None, module_name=None):
    """Import model and load pretrained weights"""

    if from_torchvision:
        try:
            import torchvision.models
            creator = getattr(torchvision.models, model_name)
            model = creator()
        except ImportError as err:
            print(err)
            sys.exit('The torchvision package was not found.'
                     'Please install it to default location or '
                     'update PYTHONPATH environment variable '
                     'with the path to the installed torchvision package.')
        except AttributeError as err:
            print('ERROR: Model {} doesn\'t exist in torchvision!'.format(model_name))
            sys.exit(err)
    else:
        sys.path.append(model_path)
        try:
            module = __import__(module_name)
            creator = getattr(module, model_name)
            model = creator()
        except ImportError as err:
            print('Module {} in {} doesn\'t exist. Check import path and name'.format(model_name, model_path))
            sys.exit(err)
        except AttributeError as err:
            print('ERROR: Module {} contains no class or function with name {}!'
                  .format(module_name, model_name))
            sys.exit(err)

    try:
        model.load_state_dict(torch.load(weights, map_location='cpu'))
    except RuntimeError as err:
        print('ERROR: Weights from \n{}\n cannot be loaded for model {}! Check matching between model and weights')
        sys.exit(err)
    return model


def convert_to_onnx(model, input_shape, output_file, input_names, output_names):
    """Convert PyTorch model to ONNX and check the resulting onnx model"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(model, dummy_input, str(output_file),
                      verbose=False, input_names=input_names, output_names=output_names)

    # Model check after conversion
    model = onnx.load(str(output_file))
    try:
        onnx.checker.check_model(model)
        print('ONNX check passed successfully.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as exc:
        sys.exit('ONNX check failed with error: ' + str(exc))


def main():
    args = parse_args()
    model = load_model(args.model_name, args.weights, args.from_torchvision, args.model_path, args.import_module)
    convert_to_onnx(model, args.input_shape, args.output_file, args.input_names, args.output_names)


if __name__ == '__main__':
    main()
