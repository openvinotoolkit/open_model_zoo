# Copyright (c) 2019 Intel Corporation
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
import contextlib
import fnmatch
import json
import re
import shlex
import shutil
import sys
import traceback

from pathlib import Path

import yaml

DOWNLOAD_TIMEOUT = 5 * 60

# make sure to update the documentation if you modify these
KNOWN_FRAMEWORKS = {'caffe', 'dldt', 'mxnet', 'pytorch', 'tf'}
KNOWN_PRECISIONS = {'FP16', 'FP32', 'INT1', 'INT8'}
KNOWN_TASK_TYPES = {
    'action_recognition',
    'classification',
    'detection',
    'face_recognition',
    'head_pose_estimation',
    'human_pose_estimation',
    'image_processing',
    'instance_segmentation',
    'object_attributes',
    'optical_character_recognition',
    'semantic_segmentation',
}

RE_MODEL_NAME = re.compile(r'[0-9a-zA-Z._-]+')
RE_SHA256SUM = re.compile(r'[0-9a-fA-F]{64}')

class Reporter:
    GROUP_DECORATION = '#' * 16 + '||'
    SECTION_DECORATION = '=' * 10
    ERROR_DECORATION = '#' * 10

    def __init__(self, enable_human_output=True, enable_json_output=False, event_context={}):
        self.enable_human_output = enable_human_output
        self.enable_json_output = enable_json_output
        self.event_context = event_context

    def print_group_heading(self, text):
        if not self.enable_human_output: return
        print(self.GROUP_DECORATION, text, self.GROUP_DECORATION[::-1])
        print()

    def print_section_heading(self, format, *args):
        if not self.enable_human_output: return
        print(self.SECTION_DECORATION, format.format(*args), flush=True)

    def print_progress(self, format, *args):
        if not self.enable_human_output: return
        print(format.format(*args), end='\r' if sys.stdout.isatty() else '\n', flush=True)

    def end_progress(self):
        if not self.enable_human_output: return
        if sys.stdout.isatty():
            print()

    def print(self, format='', *args, flush=False):
        if not self.enable_human_output: return
        print(format.format(*args), flush=flush)

    def log_warning(self, format, *args, exc_info=False):
        if exc_info:
            traceback.print_exc(file=sys.stderr)
        print(self.ERROR_DECORATION, "Warning:", format.format(*args), file=sys.stderr)

    def log_error(self, format, *args, exc_info=False):
        if exc_info:
            traceback.print_exc(file=sys.stderr)
        print(self.ERROR_DECORATION, "Error:", format.format(*args), file=sys.stderr)

    def log_details(self, format, *args):
        print(self.ERROR_DECORATION, '    ', format.format(*args), file=sys.stderr)

    def emit_event(self, type, **kwargs):
        if not self.enable_json_output: return
        json.dump({'$type': type, **self.event_context, **kwargs}, sys.stdout, indent=None)
        print()

    def with_event_context(self, **kwargs):
        return Reporter(
            enable_human_output=self.enable_human_output,
            enable_json_output=self.enable_json_output,
            event_context={**self.event_context, **kwargs},
        )

class DeserializationError(Exception):
    def __init__(self, problem, contexts=()):
        super().__init__(': '.join(contexts + (problem,)))
        self.problem = problem
        self.contexts = contexts

@contextlib.contextmanager
def deserialization_context(context):
    try:
        yield None
    except DeserializationError as exc:
        raise DeserializationError(exc.problem, (context,) + exc.contexts) from exc

def validate_string(context, value):
    if not isinstance(value, str):
        raise DeserializationError('{}: expected a string, got {!r}'.format(context, value))
    return value

def validate_string_enum(context, value, known_values):
    str_value = validate_string(context, value)
    if str_value not in known_values:
        raise DeserializationError('{}: expected one of {!r}, got {!r}'.format(context, known_values, value))
    return str_value

def validate_relative_path(context, value):
    path = Path(validate_string(context, value))

    if path.anchor or '..' in path.parts:
        raise DeserializationError('{}: disallowed absolute path or parent traversal'.format(context))

    return path

def validate_nonnegative_int(context, value):
    if not isinstance(value, int) or value < 0:
        raise DeserializationError(
            '{}: expected a non-negative integer, got {!r}'.format(context, value))
    return value

class TaggedBase:
    @classmethod
    def deserialize(cls, value):
        try:
            return cls.types[value['$type']].deserialize(value)
        except KeyError:
            raise DeserializationError('Unknown "$type": "{}"'.format(value['$type']))

class FileSource(TaggedBase):
    types = {}

    @classmethod
    def deserialize(cls, source):
        if isinstance(source, str):
            source = {'$type': 'http', 'url': source}
        return super().deserialize(source)

class FileSourceHttp(FileSource):
    def __init__(self, url):
        self.url = url

    @classmethod
    def deserialize(cls, source):
        return cls(validate_string('"url"', source['url']))

    def start_download(self, session, chunk_size):
        response = session.get(self.url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        return response.iter_content(chunk_size=chunk_size)

FileSource.types['http'] = FileSourceHttp

class FileSourceGoogleDrive(FileSource):
    def __init__(self, id):
        self.id = id

    @classmethod
    def deserialize(cls, source):
        return cls(validate_string('"id"', source['id']))

    def start_download(self, session, chunk_size):
        URL = 'https://docs.google.com/uc?export=download'
        response = session.get(URL, params={'id' : self.id}, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': self.id, 'confirm': value}
                response = session.get(URL, params=params, stream=True, timeout=DOWNLOAD_TIMEOUT)
                response.raise_for_status()

        return response.iter_content(chunk_size=chunk_size)

FileSource.types['google_drive'] = FileSourceGoogleDrive

class ModelFile:
    def __init__(self, name, size, sha256, source):
        self.name = name
        self.size = size
        self.sha256 = sha256
        self.source = source

    @classmethod
    def deserialize(cls, file):
        name = validate_relative_path('"name"', file['name'])

        with deserialization_context('In file "{}"'.format(name)):
            size = validate_nonnegative_int('"size"', file['size'])

            sha256 = validate_string('"sha256"', file['sha256'])

            if not RE_SHA256SUM.fullmatch(sha256):
                raise DeserializationError(
                    '"sha256": got invalid hash {!r}'.format(sha256))

            with deserialization_context('"source"'):
                source = FileSource.deserialize(file['source'])

            return cls(name, size, sha256, source)

class Postproc(TaggedBase):
    types = {}

class PostprocRegexReplace(Postproc):
    def __init__(self, file, pattern, replacement, count):
        self.file = file
        self.pattern = pattern
        self.replacement = replacement
        self.count = count

    @classmethod
    def deserialize(cls, postproc):
        return cls(
            validate_relative_path('"file"', postproc['file']),
            re.compile(validate_string('"pattern"', postproc['pattern'])),
            validate_string('"replacement"', postproc['replacement']),
            validate_nonnegative_int('"count"', postproc.get('count', 0)),
        )

    def apply(self, reporter, output_dir):
        postproc_file = output_dir / self.file

        reporter.print_section_heading('Replacing text in {}', postproc_file)

        postproc_file_text = postproc_file.read_text()

        orig_file = postproc_file.with_name(postproc_file.name + '.orig')
        if not orig_file.exists():
            postproc_file.replace(orig_file)

        postproc_file_text, num_replacements = self.pattern.subn(
            self.replacement, postproc_file_text, count=self.count)

        if num_replacements == 0:
            raise RuntimeError('Invalid pattern: no occurrences found')

        if self.count != 0 and num_replacements != self.count:
            raise RuntimeError('Invalid pattern: expected at least {} occurrences, but only {} found'.format(
                self.count, num_replacements))

        postproc_file.write_text(postproc_file_text)

Postproc.types['regex_replace'] = PostprocRegexReplace

class PostprocUnpackArchive(Postproc):
    def __init__(self, file, format):
        self.file = file
        self.format = format

    @classmethod
    def deserialize(cls, postproc):
        return cls(
            validate_relative_path('"file"', postproc['file']),
            validate_string('"format"', postproc['format']),
        )

    def apply(self, reporter, output_dir):
        postproc_file = output_dir / self.file

        reporter.print_section_heading('Unpacking {}', postproc_file)

        shutil.unpack_archive(str(postproc_file), str(output_dir), self.format)
        postproc_file.unlink()  # Remove the archive

Postproc.types['unpack_archive'] = PostprocUnpackArchive

class Model:
    def __init__(self, name, subdirectory, files, postprocessing, mo_args, framework,
            description, license_url, precisions, task_type, pytorch_to_onnx_args):
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
        self.pytorch_to_onnx_args = pytorch_to_onnx_args

    @classmethod
    def deserialize(cls, model, name, subdirectory):
        with deserialization_context('In model "{}"'.format(name)):
            if not RE_MODEL_NAME.fullmatch(name):
                raise DeserializationError('Invalid name, must consist only of letters, digits or ._-')

            files = []
            file_names = set()

            for file in model['files']:
                files.append(ModelFile.deserialize(file))

                if files[-1].name in file_names:
                    raise DeserializationError(
                        'Duplicate file name "{}"'.format(files[-1].name))
                file_names.add(files[-1].name)

            postprocessing = []

            for i, postproc in enumerate(model.get('postprocessing', [])):
                with deserialization_context('"postprocessing" #{}'.format(i)):
                    postprocessing.append(Postproc.deserialize(postproc))

            pytorch_to_onnx_args = None
            if model.get('pytorch_to_onnx', None):
                pytorch_to_onnx_args = [validate_string('"pytorch_to_onnx" #{}'.format(i), arg)
                                        for i, arg in enumerate(model['pytorch_to_onnx'])]

            framework = validate_string_enum('"framework"', model['framework'], KNOWN_FRAMEWORKS)

            if 'model_optimizer_args' in model:
                mo_args = [validate_string('"model_optimizer_args" #{}'.format(i), arg)
                    for i, arg in enumerate(model['model_optimizer_args'])]

                precisions = {'FP16', 'FP32'}
            else:
                if framework != 'dldt':
                    raise DeserializationError('Model not in IR format, but no conversions defined')

                mo_args = None

                files_per_precision = {}

                for file in files:
                    if len(file.name.parts) != 2:
                        raise DeserializationError('Can\'t derive precision from file name {!r}'.format(file.name))
                    p = file.name.parts[0]
                    if p not in KNOWN_PRECISIONS:
                        raise DeserializationError(
                            'Unknown precision {!r} derived from file name {!r}, expected one of {!r}'.format(
                                p, file.name, KNOWN_PRECISIONS))
                    files_per_precision.setdefault(p, set()).add(file.name.parts[1])

                for precision, precision_files in files_per_precision.items():
                    for ext in ['xml', 'bin']:
                        if (name + '.' + ext) not in precision_files:
                            raise DeserializationError('No {} file for precision "{}"'.format(ext.upper(), precision))

                precisions = set(files_per_precision.keys())

            description = validate_string('"description"', model['description'])

            license_url = validate_string('"license"', model['license'])

            task_type = validate_string_enum('"task_type"', model['task_type'], KNOWN_TASK_TYPES)

            return cls(name, subdirectory, files, postprocessing, mo_args, framework,
                description, license_url, precisions, task_type, pytorch_to_onnx_args)

def load_models(args):
    models = []
    model_names = set()

    def add_model(model):
        models.append(model)

        if models[-1].name in model_names:
            raise DeserializationError(
                'Duplicate model name "{}"'.format(models[-1].name))
        model_names.add(models[-1].name)

    if args.config is None: # per-model configs
        model_root = (Path(__file__).resolve().parent / '../../models').resolve()

        for config_path in sorted(model_root.glob('**/model.yml')):
            subdirectory = config_path.parent.relative_to(model_root)

            with config_path.open('rb') as config_file, \
                    deserialization_context('In config "{}"'.format(config_path)):

                model = yaml.safe_load(config_file)

                for bad_key in ['name', 'subdirectory']:
                    if bad_key in model:
                        raise DeserializationError('Unsupported key "{}"'.format(bad_key))

                add_model(Model.deserialize(model, subdirectory.name, subdirectory))

    else: # monolithic config
        print('########## Warning: the --config option is deprecated and will be removed in a future release',
            file=sys.stderr)
        with args.config.open('rb') as config_file, \
                deserialization_context('In config "{}"'.format(args.config)):
            for i, model in enumerate(yaml.safe_load(config_file)['topologies']):
                with deserialization_context('In model #{}'.format(i)):
                    name = validate_string('"name"', model['name'])
                    if not name: raise DeserializationError('"name": must not be empty')

                with deserialization_context('In model "{}"'.format(name)):
                    subdirectory = validate_relative_path('"output"', model['output'])

                add_model(Model.deserialize(model, name, subdirectory))

    return models

def load_models_or_die(args):
    try:
        return load_models(args)
    except DeserializationError as e:
        indent = '    '

        for i, context in enumerate(e.contexts):
            print(indent * i + context + ':', file=sys.stderr)
        print(indent * len(e.contexts) + e.problem, file=sys.stderr)
        sys.exit(1)

# requires the --print_all, --all, --name and --list arguments to be in `args`
def load_models_from_args(parser, args):
    if args.print_all:
        for model in load_models_or_die(args):
            print(model.name)
        sys.exit()

    filter_args_count = sum([args.all, args.name is not None, args.list is not None])

    if filter_args_count > 1:
        parser.error('at most one of "--all", "--name" or "--list" can be specified')

    if filter_args_count == 0:
        parser.error('one of "--print_all", "--all", "--name" or "--list" must be specified')

    all_models = load_models_or_die(args)

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
            matching_models = [model for model in all_models
                if fnmatch.fnmatchcase(model.name, pattern)]

            if not matching_models:
                sys.exit('No matching models: "{}"'.format(pattern))

            for model in matching_models:
                models[model.name] = model

        return list(models.values())
