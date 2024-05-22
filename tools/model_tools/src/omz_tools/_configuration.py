# Copyright (c) 2021-2024 Intel Corporation
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
import enum
import fnmatch
import re
import shlex
import sys
import yaml

from omz_tools import _common
from omz_tools.download_engine import cache, file_source, postprocessing, validation

RE_MODEL_NAME = re.compile(r'[0-9a-zA-Z._-]+')
EXCLUDED_MODELS = []

ModelInputInfo = collections.namedtuple('ModelInputInfo', ['name', 'shape', 'layout'])

class ModelFile:
    def __init__(self, name, size, checksum, source):
        self.name = name
        self.size = size
        self.checksum = checksum
        self.source = source

    @classmethod
    def deserialize(cls, file):
        name = validation.validate_relative_path('"name"', file['name'])

        with validation.deserialization_context('In file "{}"'.format(name)):
            size = validation.validate_nonnegative_int('"size"', file['size'])

            with validation.deserialization_context('"checksum"'):
                checksum = cache.Checksum.deserialize(file['checksum'])

            with validation.deserialization_context('"source"'):
                source = file_source.FileSource.deserialize(file['source'])

            return cls(name, size, checksum, source)

class Model:
    def __init__(
        self, name, subdirectory, files, postprocessing, mo_args, framework,
        description, license_url, precisions,
        task_type, conversion_to_onnx_args, converter_to_onnx, composite_model_name, input_info,
        model_info
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
        self.task_type = task_type
        self.model_info = model_info
        self.conversion_to_onnx_args = conversion_to_onnx_args
        self.converter_to_onnx = converter_to_onnx
        self.composite_model_name = composite_model_name
        self.input_info = input_info
        self.model_stages = {}

    @classmethod
    def deserialize(cls, model, name, subdirectory, composite_model_name, known_frameworks=None, known_task_types=None):
        if known_frameworks is None:
            known_frameworks = _common.KNOWN_FRAMEWORKS
        if known_task_types is None:
            known_task_types = _common.KNOWN_TASK_TYPES

        with validation.deserialization_context('In model "{}"'.format(name)):
            if not RE_MODEL_NAME.fullmatch(name):
                raise validation.DeserializationError('Invalid name, must consist only of letters, digits or ._-')

            files = []
            file_names = set()

            model_info = model.get('model_info', {})

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
                known_frameworks.keys())

            input_info = []
            for input in model.get('input_info', []):
                input_name = validation.validate_string('"input name"', input['name'])
                shape = validation.validate_list('"input shape"', input.get('shape', []))
                layout = validation.validate_string('"input layout"', input.get('layout', ''))
                input_info.append(ModelInputInfo(input_name, shape, layout))

            conversion_to_onnx_args = model.get('conversion_to_onnx_args', None)
            if known_frameworks[framework]:
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

            if 'model_optimizer_args' in model:
                mo_args = [validation.validate_string('"model_optimizer_args" #{}'.format(i), arg)
                    for i, arg in enumerate(model['model_optimizer_args'])]
                precisions = {'FP16', 'FP32'}
            else:
                if framework != 'dldt':
                    raise validation.DeserializationError('Model not in IR format, but no conversions defined')

                mo_args = None

                files_per_precision = {}

                for file in files:
                    if len(file.name.parts) == 2 and file.name.parts[0].startswith('FP'):
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

            description = validation.validate_string('"description"', model['description'])

            license_url = validation.validate_string('"license"', model['license'])

            task_type = validation.validate_string_enum('"task_type"', model['task_type'],
                known_task_types)

            return cls(name, subdirectory, files, postprocessings, mo_args, framework,
                description, license_url, precisions,
                task_type, conversion_to_onnx_args, known_frameworks[framework],
                composite_model_name, input_info, model_info)

class CompositeModel:
    def __init__(self, name, subdirectory, task_type, model_stages, description, framework,
        license_url, precisions, composite_model_name
    ):
        self.name = name
        self.subdirectory = subdirectory
        self.task_type = task_type
        self.model_stages = model_stages
        self.description = description
        self.framework = framework
        self.license_url = license_url
        self.precisions = precisions
        self.composite_model_name = composite_model_name
        self.input_info = []

    @classmethod
    def deserialize(cls, model, name, subdirectory, stages, known_frameworks=None, known_task_types=None):
        if known_frameworks is None:
            known_frameworks = _common.KNOWN_FRAMEWORKS
        if known_task_types is None:
            known_task_types = _common.KNOWN_TASK_TYPES

        with validation.deserialization_context('In model "{}"'.format(name)):
            if not RE_MODEL_NAME.fullmatch(name):
                raise validation.DeserializationError('Invalid name, must consist only of letters, digits or ._-')

            task_type = validation.validate_string_enum('"task_type"', model['task_type'], known_task_types)

            description = validation.validate_string('"description"', model['description'])

            license_url = validation.validate_string('"license"', model['license'])

            framework = validation.validate_string_enum('"framework"', model['framework'],
                                                        known_frameworks)

            model_stages = []
            for model_part_name, model_part in stages.items():
                model_subdirectory = model_part.get('model_subdirectory')
                model_stages.append(Model.deserialize(model_part, model_part_name, model_subdirectory, name,
                                                      known_frameworks=known_frameworks, known_task_types=known_task_types))

            precisions = model_stages[0].precisions

            return cls(name, subdirectory, task_type, model_stages, description, framework,
                license_url, precisions, name)

class ModelLoadingMode(enum.Enum):
    all = 0 # return all models
    composite_only = 1 # return only composite models
    non_composite_only = 2 # return only non composite models
    ignore_composite = 3 # ignore composite structure, return flatten models list

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

def load_models(models_root, mode=ModelLoadingMode.all):
    models = []
    model_names = set()

    composite_models = []
    composite_model_names = set()

    if mode in (ModelLoadingMode.all, ModelLoadingMode.composite_only):

        for composite_model_config in sorted(models_root.glob('**/composite-model.yml')):
            composite_model_name = composite_model_config.parent.name
            with validation.deserialization_context('In model "{}"'.format(composite_model_name)):
                if not RE_MODEL_NAME.fullmatch(composite_model_name):
                    raise validation.DeserializationError('Invalid name, must consist only of letters, digits or ._-')

                check_composite_model_dir(composite_model_config.parent)

                with composite_model_config.open('rb') as config_file, \
                    validation.deserialization_context('In config "{}"'.format(composite_model_config)):

                    composite_model = yaml.safe_load(config_file)
                    stages_order = composite_model.get('stages_order', [])
                    model_stages = {}
                    for stage in composite_model_config.parent.glob('*/model.yml'):
                        with stage.open('rb') as stage_config_file, \
                            validation.deserialization_context('In config "{}"'.format(stage_config_file)):
                            model = yaml.safe_load(stage_config_file)

                            stage_subdirectory = stage.parent.relative_to(models_root)
                            model['model_subdirectory'] = stage_subdirectory
                            model_stages[stage_subdirectory.name] = model

                    if len(model_stages) == 0:
                        continue

                    model_stages = {stage_name: model_stages[stage_name] for stage_name in stages_order}

                    subdirectory = composite_model_config.parent.relative_to(models_root)
                    composite_models.append(CompositeModel.deserialize(
                        composite_model, composite_model_name, subdirectory, model_stages
                    ))

                    if composite_model_name in composite_model_names:
                        raise validation.DeserializationError(
                            'Duplicate composite model name "{}"'.format(composite_model_name))
                    composite_model_names.add(composite_model_name)

    if mode != ModelLoadingMode.composite_only:
        for config_path in sorted(models_root.glob('**/model.yml')):
            subdirectory = config_path.parent

            is_composite = (subdirectory.parent / 'composite-model.yml').exists()
            composite_model_name = None
            if is_composite:
                if mode != ModelLoadingMode.ignore_composite:
                    continue
                composite_model_name = subdirectory.parent.name

            subdirectory = subdirectory.relative_to(models_root)

            with config_path.open('rb') as config_file, \
                    validation.deserialization_context('In config "{}"'.format(config_path)):

                model = yaml.safe_load(config_file)

                for bad_key in ['name', 'subdirectory']:
                    if bad_key in model:
                        raise validation.DeserializationError('Unsupported key "{}"'.format(bad_key))

                if subdirectory.name not in EXCLUDED_MODELS:
                    models.append(Model.deserialize(model, subdirectory.name, subdirectory, composite_model_name))
                    continue

                if models[-1].name in model_names:
                    raise validation.DeserializationError(
                        'Duplicate model name "{}"'.format(models[-1].name))
                model_names.add(models[-1].name)

    return sorted(models + composite_models, key=lambda model : model.name)

def load_models_or_die(models_root, **kwargs):
    try:
        return load_models(models_root, **kwargs)
    except validation.DeserializationError as e:
        indent = '    '

        for i, context in enumerate(e.contexts):
            print(indent * i + context + ':', file=sys.stderr)
        print(indent * len(e.contexts) + e.problem, file=sys.stderr)
        sys.exit(1)

# requires the --print_all, --all, --name and --list arguments to be in `args`
def load_models_from_args(parser, args, models_root, **kwargs):
    if args.print_all:
        for model in load_models_or_die(models_root, **kwargs):
            print(model.name)
        sys.exit()

    filter_args_count = sum([args.all, args.name is not None, args.list is not None])

    if filter_args_count > 1:
        parser.error('at most one of "--all", "--name" or "--list" can be specified')

    if filter_args_count == 0:
        parser.error('one of "--print_all", "--all", "--name" or "--list" must be specified')

    all_models = load_models_or_die(models_root)

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
            is_excluded = False
            for model in all_models:
                if fnmatch.fnmatchcase(model.name, pattern):
                    matching_models.append(model)
                elif isinstance(model, CompositeModel):
                    for model_stage in model.model_stages:
                        if fnmatch.fnmatchcase(model_stage.name, pattern):
                            matching_models.append(model_stage)

            for model in EXCLUDED_MODELS:
                if fnmatch.fnmatchcase(model, pattern):
                    is_excluded = True

            if not matching_models and not is_excluded:
                sys.exit('No matching models: "{}"'.format(pattern))

            for model in matching_models:
                models[model.name] = model

        return list(models.values())
