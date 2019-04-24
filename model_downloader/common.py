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

import contextlib
import fnmatch
import re
import shlex
import shutil
import sys

from pathlib import Path

import yaml

DOWNLOAD_TIMEOUT = 5 * 60

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
        return FileSourceHttp(validate_string('"url"', source['url']))

    def start_download(self, session, total_size=None):
        response = session.get(self.url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        if total_size is None:
            size = int(response.headers.get('content-length', 0))
        else:
            size = total_size

        return response.iter_content(chunk_size=8192), size

FileSource.types['http'] = FileSourceHttp

class FileSourceGoogleDrive(FileSource):
    def __init__(self, id):
        self.id = id

    @classmethod
    def deserialize(cls, source):
        return FileSourceGoogleDrive(validate_string('"id"', source['id']))

    def start_download(self, session, total_size):
        URL = 'https://docs.google.com/uc?export=download'
        response = session.get(URL, params={'id' : self.id}, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': self.id, 'confirm': value}
                response = session.get(URL, params=params, stream=True, timeout=DOWNLOAD_TIMEOUT)
                response.raise_for_status()

        return response.iter_content(chunk_size=32768), total_size

FileSource.types['google_drive'] = FileSourceGoogleDrive

class TopologyFile:
    def __init__(self, name, size, sha256, source):
        self.name = name
        self.size = size
        self.sha256 = sha256
        self.source = source

    @classmethod
    def deserialize(cls, file):
        name = Path(validate_string('"name"', file['name']))

        if len(name.parts) != 1 or name.anchor:
            raise DeserializationError('Invalid file name "{}"'.format(name))

        with deserialization_context('In file "{}"'):
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
        return PostprocRegexReplace(
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
        return PostprocUnpackArchive(
            validate_relative_path('"file"', postproc['file']),
            validate_string('"format"', postproc['format']),
        )

    def apply(self, output_dir):
        postproc_file = output_dir / self.file

        print('========= Unpacking {} ========='.format(postproc_file))

        shutil.unpack_archive(str(postproc_file), str(output_dir), self.format)

Postproc.types['unpack_archive'] = PostprocUnpackArchive

class Topology:
    def __init__(self, name, subdir, files, postprocessing):
        self.name = name
        self.subdir = subdir
        self.files = files
        self.postprocessing = postprocessing

    @classmethod
    def deserialize(cls, top):
        name = validate_string('"name"', top['name'])
        if not name: raise DeserializationError('"name": must not be empty')

        with deserialization_context('In topology "{}"'.format(name)):
            subdir = validate_relative_path('"output"', top['output'])

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

            return cls(name, subdir, files, postprocessing)

def load_topologies(config):
    with config.open() as config_file:
        try:
            topologies = []
            topology_names = set()

            for top in yaml.safe_load(config_file)['topologies']:
                topologies.append(Topology.deserialize(top))

                if topologies[-1].name in topology_names:
                    raise RuntimeError(
                        'In config "{}": Duplicate topology name "{}"'.format(config, topologies[-1].name))
                topology_names.add(topologies[-1].name)

            return topologies
        except DeserializationError as exc:
            raise RuntimeError('In config "{}": {}'.format(config, exc)) from exc

# requires the --print_all, --all, --name and --list arguments to be in `args`
def load_topologies_from_args(parser, args):
    if args.print_all:
        for top in load_topologies(args.config):
            print(top.name)
        sys.exit()

    filter_args_count = sum([args.all, args.name is not None, args.list is not None])

    if filter_args_count > 1:
        parser.error('at most one of "--all", "--name" or "--list" can be specified')

    if filter_args_count == 0:
        parser.error('one of "--print_all", "--all", "--name" or "--list" must be specified')

    all_topologies = load_topologies(args.config)

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

        topologies = []
        for pattern in patterns:
            matching_topologies = [top for top in all_topologies
                if fnmatch.fnmatchcase(top.name, pattern)]

            if not matching_topologies:
                sys.exit('No matching topologies: "{}"'.format(pattern))

            topologies.extend(matching_topologies)

        return topologies
# TEST
