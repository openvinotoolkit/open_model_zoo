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
import re

from pathlib import Path

import yaml

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

FileSource.types['http'] = FileSourceHttp

class FileSourceGoogleDrive(FileSource):
    def __init__(self, id):
        self.id = id

    @classmethod
    def deserialize(cls, source):
        return FileSourceGoogleDrive(validate_string('"id"', source['id']))

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
