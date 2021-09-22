# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import fnmatch
import re
import shlex
import sys
import yaml

from openvino.model_zoo import _common
from openvino.model_zoo.download_engine import file_source, postprocessing, validation

RE_MODEL_NAME = re.compile(r'[0-9a-zA-Z._-]+')
RE_SHA256SUM = re.compile(r'[0-9a-fA-F]{64}')

class ModelFile:
    def __init__(self, name, size, sha256, source):
        self.name = name
        self.size = size
        self.sha256 = sha256
        self.source = source

    @classmethod
    def deserialize(cls, file):
        name = validation.validate_relative_path('"name"', file['name'])

        with validation.deserialization_context('In file "{}"'.format(name)):
            size = validation.validate_nonnegative_int('"size"', file['size'])

            sha256_str = validation.validate_string('"sha256"', file['sha256'])

            if not RE_SHA256SUM.fullmatch(sha256_str):
                raise validation.DeserializationError(
                    '"sha256": got invalid hash {!r}'.format(sha256_str))

            sha256 = bytes.fromhex(sha256_str)

            with validation.deserialization_context('"source"'):
                source = file_source.FileSource.deserialize(file['source'])

            return cls(name, size, sha256, source)

class Model:
    def __init__(
        self, name, subdirectory, files, postprocessing, mo_args, framework,
        description, license_url, precisions, quantization_output_precisions,
        task_type, conversion_to_onnx_args, composite_model_name,
    ):
        self.name = name
        self.subdirectory = subdirectory
        self.files = files
        self.postprocessing = postprocessing
        self.mo_args = mo_args
        self.framework = framework
        self.description = description
        self.license_url = license_url
        self.precisions = precisions
        self.quantization_output_precisions = quantization_output_precisions
        self.task_type = task_type
        self.conversion_to_onnx_args = conversion_to_onnx_args
        self.converter_to_onnx = _common.KNOWN_FRAMEWORKS[framework]
        self.composite_model_name = composite_model_name

    @classmethod
    def deserialize(cls, model, name, subdirectory, composite_model_name):
        with validation.deserialization_context('In model "{}"'.format(name)):
            if not RE_MODEL_NAME.fullmatch(name):
                raise validation.DeserializationError('Invalid name, must consist only of letters, digits or ._-')

            files = []
            file_names = set()

            for file in model['files']:
                files.append(ModelFile.deserialize(file))

                if files[-1].name in file_names:
                    raise validation.DeserializationError(
                        'Duplicate file name "{}"'.format(files[-1].name))
                file_names.add(files[-1].name)

            postprocessings = []

            for i, postproc in enumerate(model.get('postprocessing', [])):
                with validation.deserialization_context('"postprocessing" #{}'.format(i)):
                    postprocessings.append(postprocessing.Postproc.deserialize(postproc))

            framework = validation.validate_string_enum('"framework"', model['framework'],
                _common.KNOWN_FRAMEWORKS.keys())

            conversion_to_onnx_args = model.get('conversion_to_onnx_args', None)
            if _common.KNOWN_FRAMEWORKS[framework]:
                if not conversion_to_onnx_args:
                    raise validation.DeserializationError('"conversion_to_onnx_args" is absent. '
                                                          'Framework "{}" is supported only by conversion to ONNX.'
                                                          .format(framework))
                conversion_to_onnx_args = [validation.validate_string('"conversion_to_onnx_args" #{}'.format(i), arg)
                                           for i, arg in enumerate(model['conversion_to_onnx_args'])]
            else:
                if conversion_to_onnx_args:
                    raise validation.DeserializationError(
                        'Conversion to ONNX not supported for "{}" framework'.format(framework))

            quantized = model.get('quantized', None)
            if quantized is not None and quantized != 'INT8':
                raise validation.DeserializationError('"quantized": expected "INT8", got {!r}'.format(quantized))

            if 'model_optimizer_args' in model:
                mo_args = [validation.validate_string('"model_optimizer_args" #{}'.format(i), arg)
                    for i, arg in enumerate(model['model_optimizer_args'])]
                precisions = {f'FP16-{quantized}', f'FP32-{quantized}'} if quantized is not None else {'FP16', 'FP32'}
            else:
                if framework != 'dldt':
                    raise validation.DeserializationError('Model not in IR format, but no conversions defined')

                mo_args = None

                files_per_precision = {}

                for file in files:
                    if len(file.name.parts) != 2:
                        raise validation.DeserializationError(
                            'Can\'t derive precision from file name {!r}'.format(file.name))
                    p = file.name.parts[0]
                    if p not in _common.KNOWN_PRECISIONS:
                        raise validation.DeserializationError(
                            'Unknown precision {!r} derived from file name {!r}, expected one of {!r}'.format(
                                p, file.name, _common.KNOWN_PRECISIONS))
                    files_per_precision.setdefault(p, set()).add(file.name.parts[1])

                for precision, precision_files in files_per_precision.items():
                    for ext in ['xml', 'bin']:
                        if (name + '.' + ext) not in precision_files:
                            raise validation.DeserializationError(
                                'No {} file for precision "{}"'.format(ext.upper(), precision))

                precisions = set(files_per_precision.keys())

            quantizable = model.get('quantizable', False)
            if not isinstance(quantizable, bool):
                raise validation.DeserializationError(
                    '"quantizable": expected a boolean, got {!r}'.format(quantizable))

            quantization_output_precisions = _common.KNOWN_QUANTIZED_PRECISIONS.keys() if quantizable else set()

            description = validation.validate_string('"description"', model['description'])

            license_url = validation.validate_string('"license"', model['license'])

            task_type = validation.validate_string_enum('"task_type"', model['task_type'],
                _common.KNOWN_TASK_TYPES)

            return cls(name, subdirectory, files, postprocessings, mo_args, framework,
                description, license_url, precisions, quantization_output_precisions,
                task_type, conversion_to_onnx_args, composite_model_name)

def check_composite_model_dir(model_dir):
    with validation.deserialization_context('In directory "{}"'.format(model_dir)):
        if list(model_dir.glob('*/*/**/model.yml')):
            raise validation.DeserializationError(
                'Directory should not contain any model.yml files in any subdirectories '
                'that are not direct children of the composite model directory')

        if (model_dir / 'model.yml').exists():
            raise validation.DeserializationError('Directory should not contain a model.yml file')

        model_name = model_dir.name
        model_stages = list(model_dir.glob('*/model.yml'))
        for model in model_stages:
            if not model.parent.name.startswith(f'{model_name}-'):
                raise validation.DeserializationError(
                    'Names of composite model parts should start with composite model name')

def load_models(models_root, args):
    models = []
    model_names = set()

    composite_models = []

    for composite_model_config in sorted(models_root.glob('**/composite-model.yml')):
        composite_model_name = composite_model_config.parent.name
        with validation.deserialization_context('In model "{}"'.format(composite_model_name)):
            if not RE_MODEL_NAME.fullmatch(composite_model_name):
                raise validation.DeserializationError('Invalid name, must consist only of letters, digits or ._-')

            check_composite_model_dir(composite_model_config.parent)

            if composite_model_name in composite_models:
                raise validation.DeserializationError(
                    'Duplicate composite model name "{}"'.format(composite_model_name))
            composite_models.append(composite_model_name)

    for config_path in sorted(models_root.glob('**/model.yml')):
        subdirectory = config_path.parent

        is_composite = (subdirectory.parent / 'composite-model.yml').exists()
        composite_model_name = subdirectory.parent.name if is_composite else None

        subdirectory = subdirectory.relative_to(models_root)

        with config_path.open('rb') as config_file, \
                validation.deserialization_context('In config "{}"'.format(config_path)):

            model = yaml.safe_load(config_file)

            for bad_key in ['name', 'subdirectory']:
                if bad_key in model:
                    raise validation.DeserializationError('Unsupported key "{}"'.format(bad_key))

            models.append(Model.deserialize(model, subdirectory.name, subdirectory, composite_model_name))

            if models[-1].name in model_names:
                raise validation.DeserializationError(
                    'Duplicate model name "{}"'.format(models[-1].name))
            model_names.add(models[-1].name)

    return models

def load_models_or_die(models_root, args):
    try:
        return load_models(models_root, args)
    except validation.DeserializationError as e:
        indent = '    '

        for i, context in enumerate(e.contexts):
            print(indent * i + context + ':', file=sys.stderr)
        print(indent * len(e.contexts) + e.problem, file=sys.stderr)
        sys.exit(1)

# requires the --print_all, --all, --name and --list arguments to be in `args`
def load_models_from_args(parser, args, models_root):
    if args.print_all:
        for model in load_models_or_die(models_root, args):
            print(model.name)
        sys.exit()

    filter_args_count = sum([args.all, args.name is not None, args.list is not None])

    if filter_args_count > 1:
        parser.error('at most one of "--all", "--name" or "--list" can be specified')

    if filter_args_count == 0:
        parser.error('one of "--print_all", "--all", "--name" or "--list" must be specified')

    all_models = load_models_or_die(models_root, args)

    if args.all:
        return all_models
    elif args.name is not None or args.list is not None:
        if args.name is not None:
            patterns = args.name.split(',')
        else:
            patterns = []
            with args.list.open() as list_file:
                for list_line in list_file:
                    tokens = shlex.split(list_line, comments=True)
                    if not tokens: continue

                    patterns.append(tokens[0])
                    # For now, ignore any other tokens in the line.
                    # We might use them as additional parameters later.

        models = collections.OrderedDict() # deduplicate models while preserving order

        for pattern in patterns:
            matching_models = []
            for model in all_models:
                if fnmatch.fnmatchcase(model.name, pattern):
                    matching_models.append(model)
                elif model.composite_model_name and fnmatch.fnmatchcase(model.composite_model_name, pattern):
                    matching_models.append(model)

            if not matching_models:
                sys.exit('No matching models: "{}"'.format(pattern))

            for model in matching_models:
                models[model.name] = model

        return list(models.values())
