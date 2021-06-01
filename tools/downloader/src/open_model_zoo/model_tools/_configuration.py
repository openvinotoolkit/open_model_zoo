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
import contextlib
import fnmatch
import re
import shlex
import shutil
import sys

from pathlib import Path

import requests
import yaml

from open_model_zoo.model_tools import _common

DOWNLOAD_TIMEOUT = 5 * 60

RE_MODEL_NAME = re.compile(r'[0-9a-zA-Z._-]+')
RE_SHA256SUM = re.compile(r'[0-9a-fA-F]{64}')

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
    RE_CONTENT_RANGE_VALUE = re.compile(r'bytes (\d+)-\d+/(?:\d+|\*)')

    types = {}

    @classmethod
    def deserialize(cls, source):
        if isinstance(source, str):
            source = {'$type': 'http', 'url': source}
        return super().deserialize(source)

    @classmethod
    def http_range_headers(cls, offset):
        if offset == 0:
            return {}

        return {
            'Accept-Encoding': 'identity',
            'Range': 'bytes={}-'.format(offset),
        }

    @classmethod
    def handle_http_response(cls, response, chunk_size):
        if response.status_code == requests.codes.partial_content:
            match = cls.RE_CONTENT_RANGE_VALUE.fullmatch(response.headers.get('Content-Range', ''))
            if not match:
                # invalid range reply; return a negative offset to make
                # the download logic restart the download.
                return None, -1

            return response.iter_content(chunk_size=chunk_size), int(match.group(1))

        # either we didn't ask for a range, or the server doesn't support ranges

        if 'Content-Range' in response.headers:
            # non-partial responses aren't supposed to have range information
            return None, -1

        return response.iter_content(chunk_size=chunk_size), 0


class FileSourceHttp(FileSource):
    def __init__(self, url):
        self.url = url

    @classmethod
    def deserialize(cls, source):
        return cls(validate_string('"url"', source['url']))

    def start_download(self, session, chunk_size, offset):
        response = session.get(self.url, stream=True, timeout=DOWNLOAD_TIMEOUT,
            headers=self.http_range_headers(offset))
        response.raise_for_status()

        return self.handle_http_response(response, chunk_size)

FileSource.types['http'] = FileSourceHttp

class FileSourceGoogleDrive(FileSource):
    def __init__(self, id):
        self.id = id

    @classmethod
    def deserialize(cls, source):
        return cls(validate_string('"id"', source['id']))

    def start_download(self, session, chunk_size, offset):
        range_headers = self.http_range_headers(offset)
        URL = 'https://docs.google.com/uc?export=download'
        response = session.get(URL, params={'id': self.id}, headers=range_headers,
            stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': self.id, 'confirm': value}
                response = session.get(URL, params=params, headers=range_headers,
                    stream=True, timeout=DOWNLOAD_TIMEOUT)
                response.raise_for_status()

        return self.handle_http_response(response, chunk_size)

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

            sha256_str = validate_string('"sha256"', file['sha256'])

            if not RE_SHA256SUM.fullmatch(sha256_str):
                raise DeserializationError(
                    '"sha256": got invalid hash {!r}'.format(sha256_str))

            sha256 = bytes.fromhex(sha256_str)

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

        postproc_file_text = postproc_file.read_text(encoding='utf-8')

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

        postproc_file.write_text(postproc_file_text, encoding='utf-8')

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

        shutil.unpack_archive(str(postproc_file), str(postproc_file.parent), self.format)
        postproc_file.unlink()  # Remove the archive

Postproc.types['unpack_archive'] = PostprocUnpackArchive

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

            framework = validate_string_enum('"framework"', model['framework'],
                _common.KNOWN_FRAMEWORKS.keys())

            conversion_to_onnx_args = model.get('conversion_to_onnx_args', None)
            if _common.KNOWN_FRAMEWORKS[framework]:
                if not conversion_to_onnx_args:
                    raise DeserializationError('"conversion_to_onnx_args" is absent. '
                                               'Framework "{}" is supported only by conversion to ONNX.'
                                               .format(framework))
                conversion_to_onnx_args = [validate_string('"conversion_to_onnx_args" #{}'.format(i), arg)
                                           for i, arg in enumerate(model['conversion_to_onnx_args'])]
            else:
                if conversion_to_onnx_args:
                    raise DeserializationError('Conversion to ONNX not supported for "{}" framework'.format(framework))

            quantized = model.get('quantized', None)
            if quantized is not None and quantized != 'INT8':
                raise DeserializationError('"quantized": expected "INT8", got {!r}'.format(quantized))

            if 'model_optimizer_args' in model:
                mo_args = [validate_string('"model_optimizer_args" #{}'.format(i), arg)
                    for i, arg in enumerate(model['model_optimizer_args'])]
                precisions = {f'FP16-{quantized}', f'FP32-{quantized}'} if quantized is not None else {'FP16', 'FP32'}
            else:
                if framework != 'dldt':
                    raise DeserializationError('Model not in IR format, but no conversions defined')

                mo_args = None

                files_per_precision = {}

                for file in files:
                    if len(file.name.parts) != 2:
                        raise DeserializationError('Can\'t derive precision from file name {!r}'.format(file.name))
                    p = file.name.parts[0]
                    if p not in _common.KNOWN_PRECISIONS:
                        raise DeserializationError(
                            'Unknown precision {!r} derived from file name {!r}, expected one of {!r}'.format(
                                p, file.name, _common.KNOWN_PRECISIONS))
                    files_per_precision.setdefault(p, set()).add(file.name.parts[1])

                for precision, precision_files in files_per_precision.items():
                    for ext in ['xml', 'bin']:
                        if (name + '.' + ext) not in precision_files:
                            raise DeserializationError('No {} file for precision "{}"'.format(ext.upper(), precision))

                precisions = set(files_per_precision.keys())

            quantizable = model.get('quantizable', False)
            if not isinstance(quantizable, bool):
                raise DeserializationError('"quantizable": expected a boolean, got {!r}'.format(quantizable))

            quantization_output_precisions = _common.KNOWN_QUANTIZED_PRECISIONS.keys() if quantizable else set()

            description = validate_string('"description"', model['description'])

            license_url = validate_string('"license"', model['license'])

            task_type = validate_string_enum('"task_type"', model['task_type'],
                _common.KNOWN_TASK_TYPES)

            return cls(name, subdirectory, files, postprocessing, mo_args, framework,
                description, license_url, precisions, quantization_output_precisions,
                task_type, conversion_to_onnx_args, composite_model_name)

def check_composite_model_dir(model_dir):
    with deserialization_context('In directory "{}"'.format(model_dir)):
        if list(model_dir.glob('*/*/**/model.yml')):
            raise DeserializationError(
                'Directory should not contain any model.yml files in any subdirectories '
                'that are not direct children of the composite model directory')

        if (model_dir / 'model.yml').exists():
            raise DeserializationError('Directory should not contain a model.yml file')

        model_name = model_dir.name
        model_stages = list(model_dir.glob('*/model.yml'))
        for model in model_stages:
            if not model.parent.name.startswith(f'{model_name}-'):
                raise DeserializationError('Names of composite model parts should start with composite model name')

def load_models(args):
    models = []
    model_names = set()

    composite_models = []

    for composite_model_config in sorted(_common.MODEL_ROOT.glob('**/composite-model.yml')):
        composite_model_name = composite_model_config.parent.name
        with deserialization_context('In model "{}"'.format(composite_model_name)):
            if not RE_MODEL_NAME.fullmatch(composite_model_name):
                raise DeserializationError('Invalid name, must consist only of letters, digits or ._-')

            check_composite_model_dir(composite_model_config.parent)

            if composite_model_name in composite_models:
                raise DeserializationError(
                    'Duplicate composite model name "{}"'.format(composite_model_name))
            composite_models.append(composite_model_name)

    for config_path in sorted(_common.MODEL_ROOT.glob('**/model.yml')):
        subdirectory = config_path.parent

        is_composite = (subdirectory.parent / 'composite-model.yml').exists()
        composite_model_name = subdirectory.parent.name if is_composite else None

        subdirectory = subdirectory.relative_to(_common.MODEL_ROOT)

        with config_path.open('rb') as config_file, \
                deserialization_context('In config "{}"'.format(config_path)):

            model = yaml.safe_load(config_file)

            for bad_key in ['name', 'subdirectory']:
                if bad_key in model:
                    raise DeserializationError('Unsupported key "{}"'.format(bad_key))

            models.append(Model.deserialize(model, subdirectory.name, subdirectory, composite_model_name))

            if models[-1].name in model_names:
                raise DeserializationError(
                    'Duplicate model name "{}"'.format(models[-1].name))
            model_names.add(models[-1].name)

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
