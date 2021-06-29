import argparse
import ast
import importlib
import os
import sys

from pathlib import Path

import onnx
import torch
import torch.onnx


INPUT_DTYPE_TO_TORCH = {
    'bool': torch.bool,
    'double': torch.double,
    'float': torch.float,
    'half': torch.half,
    'int32': torch.int32,
    'int8': torch.int8,
    'long': torch.long,
    'short': torch.short,
    'uint8': torch.uint8,
}


def is_sequence(element):
    return isinstance(element, (list, tuple))


def shapes_arg(values):
    """Checks that the argument represents a tensor shape or a sequence of tensor shapes"""
    shapes = ast.literal_eval(values)
    if not is_sequence(shapes):
        raise argparse.ArgumentTypeError('{!r}: must be a sequence'.format(shapes))
    if not all(is_sequence(shape) for shape in shapes):
        shapes = (shapes, )
    for shape in shapes:
        if not is_sequence(shape):
            raise argparse.ArgumentTypeError('{!r}: must be a sequence'.format(shape))
        for value in shape:
            if not isinstance(value, int) or value < 0:
                raise argparse.ArgumentTypeError('Argument {!r} must be a positive integer'.format(value))
    return shapes


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
    parser.add_argument('--weights', type=str,
                        help='Path to the weights in PyTorch\'s format')
    parser.add_argument('--input-shapes', metavar='SHAPE[,SHAPE...]', type=shapes_arg, required=True,
                        help='Comma-separated shapes of the input blobs. Example: [1,1,256,256],[1,3,256,256],...')
    parser.add_argument('--output-file', type=Path, required=True,
                        help='Path to the output ONNX model')
    parser.add_argument('--model-path', type=str, action='append', dest='model_paths',
                        help='Path to PyTorch model\'s source code')
    parser.add_argument('--import-module', type=str, required=True,
                        help='Name of module, which contains model\'s constructor')
    parser.add_argument('--input-names', type=str, metavar='L[,L...]', required=True,
                        help='Comma-separated names of the input layers')
    parser.add_argument('--output-names', type=str, metavar='L[,L...]', required=True,
                        help='Comma-separated names of the output layers')
    parser.add_argument('--model-param', type=model_parameter, default=[], action='append',
                        help='Pair "name"="value" of model constructor parameter')
    parser.add_argument('--inputs-dtype', type=str, required=False, choices=INPUT_DTYPE_TO_TORCH, default='float',
                        help='Data type for inputs')
    return parser.parse_args()


def load_model(model_name, weights, model_paths, module_name, model_params):
    """Import model and load pretrained weights"""

    if model_paths:
        sys.path.extend(model_paths)

    try:
        module = importlib.import_module(module_name)
        creator = getattr(module, model_name)
        model = creator(**model_params)
    except ImportError as err:
        if model_paths:
            print('Module {} in {} doesn\'t exist. Check import path and name'.format(
                model_name, os.pathsep.join(model_paths)))
        else:
            print('Module {} doesn\'t exist. Check if it is installed'.format(model_name))
        sys.exit(err)
    except AttributeError as err:
        print('ERROR: Module {} contains no class or function with name {}!'
              .format(module_name, model_name))
        sys.exit(err)

    try:
        if weights:
            model.load_state_dict(torch.load(weights, map_location='cpu'))
    except RuntimeError as err:
        print('ERROR: Weights from {} cannot be loaded for model {}! Check matching between model and weights'.format(
            weights, model_name))
        sys.exit(err)
    return model


@torch.no_grad()
def convert_to_onnx(model, input_shapes, output_file, input_names, output_names, inputs_dtype):
    """Convert PyTorch model to ONNX and check the resulting onnx model"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_inputs = tuple(
        torch.zeros(input_shape, dtype=INPUT_DTYPE_TO_TORCH[inputs_dtype])
        for input_shape in input_shapes)
    model(*dummy_inputs)
    torch.onnx.export(model, dummy_inputs, str(output_file), verbose=False, opset_version=11,
                      input_names=input_names.split(','), output_names=output_names.split(','))

    model = onnx.load(str(output_file))

    try:
        onnx.checker.check_model(model)
        print('ONNX check passed successfully.')
    except onnx.onnx_cpp2py_export.checker.ValidationError as exc:
        sys.exit('ONNX check failed with error: ' + str(exc))


def main():
    args = parse_args()
    model = load_model(args.model_name, args.weights,
                       args.model_paths, args.import_module, dict(args.model_param))

    convert_to_onnx(model, args.input_shapes, args.output_file, args.input_names, args.output_names, args.inputs_dtype)


if __name__ == '__main__':
    main()
