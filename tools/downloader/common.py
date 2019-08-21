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
import re
import shlex
import shutil
import sys

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

class DeserializationError(Exception):
    pass

@contextlib.contextmanager
def deserialization_context(message):
    try:
        yield None
    except DeserializationError as exc:
        raise DeserializationError('{}: {}'.format(message, exc)) from exc

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

    def start_download(self, session, chunk_size, total_size=None):
        response = session.get(self.url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        if total_size is None:
            size = int(response.headers.get('content-length', 0))
        else:
            size = total_size

        return response.iter_content(chunk_size=chunk_size), size

FileSource.types['http'] = FileSourceHttp

class FileSourceGoogleDrive(FileSource):
    def __init__(self, id):
        self.id = id

    @classmethod
    def deserialize(cls, source):
        return cls(validate_string('"id"', source['id']))

    def start_download(self, session, chunk_size, total_size):
        URL = 'https://docs.google.com/uc?export=download'
        response = session.get(URL, params={'id' : self.id}, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': self.id, 'confirm': value}
                response = session.get(URL, params=params, stream=True, timeout=DOWNLOAD_TIMEOUT)
                response.raise_for_status()

        return response.iter_content(chunk_size=chunk_size), total_size

FileSource.types['google_drive'] = FileSourceGoogleDrive

class TopologyFile:
    def __init__(self, name, size, sha256, source):
        self.name = name
        self.size = size
        self.sha256 = sha256
        self.source = source

    @classmethod
    def deserialize(cls, file):
        name = validate_relative_path('"name"', file['name'])

        with deserialization_context('In file "{}"'.format(name)):
            size = file.get('size')
            if size is not None:
                size = validate_nonnegative_int('"size"', size)

            sha256 = validate_string('"sha256"', file['sha256'])

            if not re.fullmatch(r'[0-9a-fA-F]{64}', sha256):
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

    def apply(self, output_dir):
        postproc_file = output_dir / self.file

        print('========= Replacing text in {} ========='.format(postproc_file))

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

    def apply(self, output_dir):
        postproc_file = output_dir / self.file

        print('========= Unpacking {} ========='.format(postproc_file))

        shutil.unpack_archive(str(postproc_file), str(output_dir), self.format)

Postproc.types['unpack_archive'] = PostprocUnpackArchive

class Topology:
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
    def deserialize(cls, top, name, subdirectory):
        with deserialization_context('In topology "{}"'.format(name)):
            files = []
            file_names = set()

            for file in top['files']:
                files.append(TopologyFile.deserialize(file))

                if files[-1].name in file_names:
                    raise DeserializationError(
                        'Duplicate file name "{}"'.format(files[-1].name))
                file_names.add(files[-1].name)

            postprocessing = []

            for i, postproc in enumerate(top.get('postprocessing', [])):
                with deserialization_context('"postprocessing" #{}'.format(i)):
                    postprocessing.append(Postproc.deserialize(postproc))

            pytorch_to_onnx_args = None
            if top.get('pytorch_to_onnx', None):
                pytorch_to_onnx_args = [validate_string('"pytorch_to_onnx" #{}'.format(i), arg)
                                        for i, arg in enumerate(top['pytorch_to_onnx'])]


            if 'model_optimizer_args' in top:
                mo_args = [validate_string('"model_optimizer_args" #{}'.format(i), arg)
                    for i, arg in enumerate(top['model_optimizer_args'])]

                precisions = {'FP16', 'FP32'}
            else:
                mo_args = None

                def file_precision(file):
                    if len(file.name.parts) < 2:
                        raise DeserializationError('Can\'t derive precision from file name {!r}'.format(file.name))
                    p = file.name.parts[0]
                    if p not in KNOWN_PRECISIONS:
                        raise DeserializationError(
                            'Unknown precision {!r} derived from file name {!r}, expected one of {!r}'.format(
                                p, file.name, KNOWN_PRECISIONS))
                    return p

                precisions = set(map(file_precision, files))

            framework = validate_string_enum('"framework"', top['framework'], KNOWN_FRAMEWORKS)

            description = validate_string('"description"', top['description'])

            license_url = validate_string('"license"', top['license'])

            task_type = validate_string_enum('"task_type"', top['task_type'], KNOWN_TASK_TYPES)

            return cls(name, subdirectory, files, postprocessing, mo_args, framework,
                description, license_url, precisions, task_type, pytorch_to_onnx_args)

def load_topologies(args):
    topologies = []
    topology_names = set()

    def add_topology(topology):
        topologies.append(topology)

        if topologies[-1].name in topology_names:
            raise DeserializationError(
                'Duplicate topology name "{}"'.format(topologies[-1].name))
        topology_names.add(topologies[-1].name)

    if args.config is None: # per-model configs
        model_root = (Path(__file__).resolve().parent / '../../models').resolve()

        for config_path in sorted(model_root.glob('**/model.yml')):
            subdirectory = config_path.parent.relative_to(model_root)

            with config_path.open('rb') as config_file, \
                    deserialization_context('In config "{}"'.format(config_path)):

                top = yaml.safe_load(config_file)

                for bad_key in ['name', 'subdirectory']:
                    if bad_key in top:
                        raise DeserializationError('Unsupported key "{}"'.format(bad_key))

                add_topology(Topology.deserialize(top, subdirectory.name, subdirectory))

    else: # monolithic config
        print('########## Warning: the --config option is deprecated and will be removed in a future release',
            file=sys.stderr)
        with args.config.open('rb') as config_file, \
                deserialization_context('In config "{}"'.format(args.config)):
            for i, top in enumerate(yaml.safe_load(config_file)['topologies']):
                with deserialization_context('In topology #{}'.format(i)):
                    name = validate_string('"name"', top['name'])
                    if not name: raise DeserializationError('"name": must not be empty')

                with deserialization_context('In topology "{}"'.format(name)):
                    subdirectory = validate_relative_path('"output"', top['output'])

                add_topology(Topology.deserialize(top, name, subdirectory))

    return topologies

# requires the --print_all, --all, --name and --list arguments to be in `args`
def load_topologies_from_args(parser, args):
    if args.print_all:
        for top in load_topologies(args):
            print(top.name)
        sys.exit()

    filter_args_count = sum([args.all, args.name is not None, args.list is not None])

    if filter_args_count > 1:
        parser.error('at most one of "--all", "--name" or "--list" can be specified')

    if filter_args_count == 0:
        parser.error('one of "--print_all", "--all", "--name" or "--list" must be specified')

    all_topologies = load_topologies(args)

    if args.all:
        return all_topologies
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

        topologies = collections.OrderedDict() # deduplicate topologies while preserving order

        for pattern in patterns:
            matching_topologies = [top for top in all_topologies
                if fnmatch.fnmatchcase(top.name, pattern)]

            if not matching_topologies:
                sys.exit('No matching topologies: "{}"'.format(pattern))

            for top in matching_topologies:
                topologies[top.name] = top

        return list(topologies.values())
