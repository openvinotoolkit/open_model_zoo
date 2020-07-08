import argparse
import importlib
import sys

from pathlib import Path

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


def model_parameter(parameter):
    param, value = parameter.split('=', 1)
    try:
        value = eval(value, {}, {})
    except NameError as err:
        print('Cannot evaluate {!r} value in {}. For string values use "{}=\'{}\'" (with all quotes).'
              .format(value, parameter, param, value))
        sys.exit(err)
    return param, value


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
    parser.add_argument('--model-path', type=str,
                        help='Path to PyTorch model\'s source code')
    parser.add_argument('--import-module', type=str, required=True,
                        help='Name of module, which contains model\'s constructor')
    parser.add_argument('--input-names', type=str, metavar='L[,L...]',
                        help='Space separated names of the input layers')
    parser.add_argument('--output-names', type=str, metavar='L[,L...]',
                        help='Space separated names of the output layers')
    parser.add_argument('--model-param', type=model_parameter, default=[], action='append',
                        help='Pair "name"="value" of model constructor parameter')
    return parser.parse_args()


def load_model(model_name, weights, model_path, module_name, model_params):
    """Import model and load pretrained weights"""

    if model_path:
        sys.path.append(model_path)

    try:
        module = importlib.import_module(module_name)
        creator = getattr(module, model_name)
        model = creator(**model_params)
    except ImportError as err:
        if model_path:
            print('Module {} in {} doesn\'t exist. Check import path and name'.format(model_name, model_path))
        else:
            print('Module {} doesn\'t exist. Check if it is installed'.format(model_name))
        sys.exit(err)
    except AttributeError as err:
        print('ERROR: Module {} contains no class or function with name {}!'
              .format(module_name, model_name))
        sys.exit(err)

    try:
        model.load_state_dict(torch.load(weights, map_location='cpu'))
    except RuntimeError as err:
        print('ERROR: Weights from {} cannot be loaded for model {}! Check matching between model and weights'.format(
            weights, model_name))
        sys.exit(err)
    return model


def convert_to_onnx(model, input_shape, output_file, input_names, output_names):
    """Convert PyTorch model to ONNX and check the resulting onnx model"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(input_shape)
    model(dummy_input)
    torch.onnx.export(model, dummy_input, str(output_file), verbose=False, opset_version=9,
                      input_names=input_names.split(','), output_names=output_names.split(','))

    model = onnx.load(str(output_file))

    # Model Optimizer takes output names from ONNX node names if they exist.
    # However, the names PyTorch assigns to the ONNX nodes are generic and
    # non-descriptive (e.g. "Gemm_151"). By deleting these names, we make
    # MO fall back to the ONNX output names, which we can set to whatever we want.
    for node in model.graph.node:
        node.ClearField('name')

    try:
        onnx.checker.check_model(model)
        print('ONNX check passed successfully.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as exc:
        sys.exit('ONNX check failed with error: ' + str(exc))

    onnx.save(model, str(output_file))

def main():
    args = parse_args()
    model = load_model(args.model_name, args.weights,
                       args.model_path, args.import_module, dict(args.model_param))

    convert_to_onnx(model, args.input_shape, args.output_file, args.input_names, args.output_names)


if __name__ == '__main__':
    main()
