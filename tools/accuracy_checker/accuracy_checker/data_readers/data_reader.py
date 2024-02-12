"""
Copyright (c) 2018-2024 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import re
from collections import OrderedDict, namedtuple
from functools import singledispatch
from pathlib import Path

import numpy as np

from ..utils import (
    get_path, get_parameter_value_from_config
)
from ..dependency import ClassProvider, UnregisteredProviderException
from ..config import (
    BaseField, StringField, ConfigValidator, ConfigError, DictField, BoolField, PathField
)

REQUIRES_ANNOTATIONS = ['annotation_features_extractor', ]
DOES_NOT_REQUIRED_DATA_SOURCE = REQUIRES_ANNOTATIONS + ['ncf_reader']
DATA_SOURCE_IS_FILE = ['opencv_capture']


class DataRepresentation:
    def __init__(self, data, meta=None, identifier=''):
        self.identifier = identifier
        self.data = data
        self.metadata = meta or {}
        if np.isscalar(data):
            self.metadata['image_size'] = 1
        elif isinstance(data, list) and np.isscalar(data[0]):
            self.metadata['image_size'] = len(data)
        elif isinstance(data, dict):
            self.metadata['image_size'] = data.values().next().shape
        else:
            self.metadata['image_size'] = data.shape if not isinstance(data, list) else np.shape(data[0])


class AnnotationDataIdentifier:
    def __init__(self, ann_id, data_id):
        self.annotation_id = ann_id
        self.data_id = data_id

ClipIdentifier = namedtuple('ClipIdentifier', ['video', 'clip_id', 'frames'])
MultiFramesInputIdentifier = namedtuple('MultiFramesInputIdentifier', ['input_id', 'frames'])
ImagePairIdentifier = namedtuple('ImagePairIdentifier', ['first', 'second'])
ListIdentifier = namedtuple('ListIdentifier', ['values'])
MultiInstanceIdentifier = namedtuple('MultiInstanceIdentifier', ['identifier', 'object_id'])
KaldiMatrixIdentifier = namedtuple('KaldiMatrixIdentifier', ['file', 'key'])
KaldiFrameIdentifier = namedtuple('KaldiFrameIdentifier', ['file', 'key', 'id'])
ParametricImageIdentifier = namedtuple('ParametricImageIdentifier', ['identifier', 'parameters'])
VideoFrameIdentifier = namedtuple('VideoFrameIdentifier', ['video_id', 'frame'])

IdentifierSerializationOptions = namedtuple(
    "identifierSerializationOptions", ['type', 'fields', 'class_id', 'recursive', 'to_tuple']
)

identifier_serialization = {
    'AnnotationDataIdentifier': IdentifierSerializationOptions(
        'annotation_data_identifier', ['annotation_id', 'data_id'],
        AnnotationDataIdentifier, [False, True], [False, True]
    ),
    'ClipIdentifier': IdentifierSerializationOptions(
        'clip_identifier', ['video', 'clip_id', 'frames'], ClipIdentifier, [False, False, False], [False, False, True]),
    'MultiFramesInputIdentifier': IdentifierSerializationOptions(
        'multi_frame_identifier', ['input_id', 'frames'], MultiFramesInputIdentifier, [False, False], [False, True]),
    'ImagePairIdentifier': IdentifierSerializationOptions(
        'image_pair_identifier', ['first', 'second'], ImagePairIdentifier, False, False),
    'ListIdentifier': IdentifierSerializationOptions('list_identifier', ['values'], ListIdentifier, [True], [True]),
    'MultiInstanceIdentifier': IdentifierSerializationOptions(
        'multi_instance', ['identifier', 'instance_id'], MultiInstanceIdentifier, [False, False], [False, False]),
    'KaldiMatrixIdentifier': IdentifierSerializationOptions(
        'kaldi_matrix', ['file', 'key'], KaldiMatrixIdentifier, [False, False], [False, False]),
    'KaldiFrameIdentifier': IdentifierSerializationOptions(
        'kaldi_frame', ['file', 'key', 'id'], KaldiFrameIdentifier, [False, False, False], [False, False, False]),
    'ParametricImageIdentifier': IdentifierSerializationOptions(
        'parametric_image_identifier', ['identifier', 'parameters'], ParametricImageIdentifier, False, [False, True]
    ),
    'VideoFrameIdentifier':
        IdentifierSerializationOptions('video_frame', ['video_id', 'frame'], VideoFrameIdentifier, False, False)
}

identifier_deserialization = {option.type: option for option in identifier_serialization.values()}


def serialize_identifier(identifier):
    if type(identifier).__name__ in identifier_serialization:
        options = identifier_serialization[type(identifier).__name__]
        serialized_id = {'type': options.type}
        for idx, field in options.fields:
            serialized_id[field] = (
                identifier[idx] if not options.recursive[idx] else serialize_identifier(identifier[idx])
            )
        return serialized_id

    return identifier


def deserialize_identifier(identifier):
    if isinstance(identifier, dict):
        options = identifier_deserialization.get(identifier.get('type'))
        if options is None:
            raise ValueError('Unsupported identifier type: {}'.format(identifier.get('type')))
        fields = []
        for field, recursive, to_tuple in zip(options.fields, options.recursive, options.to_tuple):
            if field not in identifier:
                raise ValueError('Field {} required for identifier deserialization'.format(field))
            data = identifier[field]
            if recursive:
                data = deserialize_identifier(data)
            if to_tuple:
                data = tuple(data)
            fields.append(data)
        return options.class_id(*fields)

    return identifier


def create_ann_identifier_key(identifier):
    if isinstance(identifier, list):
        return ListIdentifier(tuple(create_ann_identifier_key(elem) for elem in identifier))
    if isinstance(identifier, ClipIdentifier):
        return ClipIdentifier(identifier.video, identifier.clip_id, tuple(identifier.frames))
    if isinstance(identifier, MultiFramesInputIdentifier):
        return MultiFramesInputIdentifier(tuple(identifier.input_id), tuple(identifier.frames))
    if isinstance(identifier, ParametricImageIdentifier):
        return ParametricImageIdentifier(identifier.identifier, tuple(identifier.parameters))
    if isinstance(identifier, AnnotationDataIdentifier):
        return identifier.annotation_id
    return identifier


def create_identifier_key(identifier):
    if isinstance(identifier, list):
        return ListIdentifier(tuple(create_ann_identifier_key(elem) for elem in identifier))
    if isinstance(identifier, ClipIdentifier):
        return ClipIdentifier(identifier.video, identifier.clip_id, tuple(identifier.frames))
    if isinstance(identifier, MultiFramesInputIdentifier):
        return MultiFramesInputIdentifier(tuple(identifier.input_id), tuple(identifier.frames))
    if isinstance(identifier, ParametricImageIdentifier):
        return ParametricImageIdentifier(identifier.identifier, tuple(identifier.parameters))
    if isinstance(identifier, AnnotationDataIdentifier):
        return AnnotationDataIdentifier(identifier.annotation_id, tuple(identifier.data_id))
    return identifier


def create_reader(config):
    return BaseReader.provide(config.get('type', 'opencv_imread'), config.get('data_source'), config=config)


class DataReaderField(BaseField):
    def validate(self, entry, field_uri=None, fetch_only=False, validation_scheme=None):
        errors = super().validate(entry, field_uri)

        if entry is None:
            return errors

        field_uri = field_uri or self.field_uri
        if isinstance(entry, str):
            errors.extend(
                StringField(choices=BaseReader.providers).validate(
                    entry, field_uri, fetch_only=fetch_only, validation_scheme=validation_scheme)
            )
        elif isinstance(entry, dict):
            class DictReaderValidator(ConfigValidator):
                type = StringField(choices=BaseReader.providers)

            dict_reader_validator = DictReaderValidator(
                'reader', on_extra_argument=DictReaderValidator.IGNORE_ON_EXTRA_ARGUMENT
            )
            errors.extend(
                dict_reader_validator.validate(
                    entry, field_uri, fetch_only=fetch_only, validation_scheme=validation_scheme
                ))
        else:
            msg = 'reader must be either string or dictionary'
            if not fetch_only:
                self.raise_error(entry, field_uri, msg)
            errors.append(self.build_error(entry, field_uri, msg, validation_scheme))

        return errors


class BaseReader(ClassProvider):
    __provider_type__ = 'reader'

    def __init__(self, data_source, config=None, postpone_data_source=False, **kwargs):
        self.config = config or {}
        self._postpone_data_source = postpone_data_source
        self.data_source = data_source
        self.read_dispatcher = singledispatch(self.read)
        self.read_dispatcher.register(list, self._read_list)
        self.read_dispatcher.register(ClipIdentifier, self._read_clip)
        self.read_dispatcher.register(MultiFramesInputIdentifier, self._read_frames_multi_input)
        self.read_dispatcher.register(ImagePairIdentifier, self._read_pair)
        self.read_dispatcher.register(ListIdentifier, self._read_list_ids)
        self.read_dispatcher.register(MultiInstanceIdentifier, self._read_multi_instance_single_object)
        self.read_dispatcher.register(ParametricImageIdentifier, self._read_parametric_input)
        self.read_dispatcher.register(VideoFrameIdentifier, self._read_video_frame)
        self.multi_infer = False

        self.validate_config(config, data_source)
        self.configure()

    def __call__(self, identifier):
        return self.read_item(identifier)

    @classmethod
    def parameters(cls):
        return {
            'type': StringField(
                default=cls.__provider__ if hasattr(cls, '__provider__') else None, description='Reader type.'
            ),
            'multi_infer': BoolField(
                default=False, optional=True, description='Allows multi infer.'
            ),
            'data_layout': StringField(optional=True, description='data layout after reading')
        }

    def get_value_from_config(self, key):
        return get_parameter_value_from_config(self.config, self.parameters(), key)

    def configure(self):
        if not self.data_source:
            if not self._postpone_data_source:
                raise ConfigError('data_source parameter is required to create "{}" '
                                  'data reader and read data'.format(self.__provider__))
        else:
            self.data_source = get_path(self.data_source, is_directory=True)
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.data_layout = self.get_value_from_config('data_layout')

    @classmethod
    def validate_config(
            cls, config, data_source=None, fetch_only=False, check_data_source=True, check_reader_type=False, **kwargs
    ):
        uri_prefix = kwargs.pop('uri_prefix', '')
        reader_uri = uri_prefix or 'reader'
        if cls.__name__ == BaseReader.__name__:
            errors = []
            reader_type = config if isinstance(config, str) else config.get('type')
            if not reader_type:
                error = ConfigError(
                    'type is not provided', config, reader_uri, validation_scheme=cls.validation_scheme()
                )
                if not fetch_only:
                    raise error
                errors.append(error)
                return errors
            try:
                reader_cls = cls.resolve(reader_type)
                reader_config = config if isinstance(config, dict) else {'type': reader_type}
                if reader_type not in DOES_NOT_REQUIRED_DATA_SOURCE and check_data_source:
                    data_source_field = PathField(
                        is_directory=reader_type not in DATA_SOURCE_IS_FILE, description='data source'
                    )
                    errors.extend(
                        data_source_field.validate(
                            data_source, reader_uri.replace('reader', 'data_source'), fetch_only=fetch_only,
                            validation_scheme=data_source_field
                        )
                    )
                errors.extend(
                    reader_cls.validate_config(reader_config, fetch_only=fetch_only, **kwargs, uri_prefix=uri_prefix))
                return errors
            except UnregisteredProviderException as exception:
                if not fetch_only:
                    raise exception
                if check_reader_type:
                    error = ConfigError('Invalid value "{}" for {}'.format(reader_type, reader_uri),
                                        config, reader_uri, validation_scheme=cls.validation_scheme())
                    errors.append(error)
                return errors
        if 'on_extra_argument' not in kwargs:
            kwargs['on_extra_argument'] = ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
        return ConfigValidator(reader_uri, fields=cls.parameters(), **kwargs).validate(
            config or {}, fetch_only=fetch_only, validation_scheme=cls.validation_scheme()
        )

    def read(self, data_id):
        raise NotImplementedError

    def _read_list(self, data_id):
        return [self.read(identifier) for identifier in data_id]

    def _read_list_ids(self, data_id):
        return self.read_dispatcher(list(data_id.values))

    def _read_clip(self, data_id):
        video = Path(data_id.video)
        frames_identifiers = [video / frame for frame in data_id.frames]
        return self.read_dispatcher(frames_identifiers)

    def _read_frames_multi_input(self, data_id):
        return self.read_dispatcher(list(data_id.frames))

    def _read_multi_instance_single_object(self, data_id):
        return self.read_dispatcher(data_id.identifier)

    def _read_parametric_input(self, data_id):
        data = self.read_dispatcher(data_id.identifier)
        return [data, *data_id.parameters]

    def _read_video_frame(self, data_id):
        return self.read_dispatcher(data_id.frame)

    def read_item(self, data_id):
        data_rep = DataRepresentation(
            self.read_dispatcher(data_id),
            identifier=data_id if not isinstance(data_id, ListIdentifier) else list(data_id.values)
        )
        if self.multi_infer:
            data_rep.metadata['multi_infer'] = True
        if self.data_layout:
            data_rep.metadata['data_layout'] = self.data_layout
        return data_rep

    def _read_pair(self, data_id):
        data = self.read_dispatcher([data_id.first, data_id.second])
        return data

    @property
    def name(self):
        return self.__provider__

    def reset(self):
        pass

    @classmethod
    def validation_scheme(cls, provider=None):
        if cls.__name__ == BaseReader.__name__:
            if provider:
                return cls.resolve(provider).validation_scheme()
            full_scheme = {}
            for provider_ in cls.providers:
                full_scheme[provider_] = cls.resolve(provider_).validation_scheme()
            return full_scheme
        return cls.parameters()


class ReaderCombiner(BaseReader):
    __provider__ = 'combine_reader'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'scheme': DictField(value_type=DataReaderField(), key_type=StringField(), allow_empty=False,
                                description='Dictionary for describing reading scheme which depends on file names.')
        })
        return parameters

    def configure(self):
        scheme = self.get_value_from_config('scheme')
        reading_scheme = OrderedDict()
        for pattern, reader_config in scheme.items():
            reader_type = reader_config['type'] if isinstance(reader_config, dict) else reader_config
            reader_configuration = reader_config if isinstance(reader_config, dict) else None
            reader = BaseReader.provide(reader_type, self.data_source, reader_configuration)
            pattern = re.compile(pattern)
            reading_scheme[pattern] = reader

        self.reading_scheme = reading_scheme
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.data_layout = self.get_value_from_config('data_layout')

    def read(self, data_id):
        for pattern, reader in self.reading_scheme.items():
            if pattern.match(str(data_id)):
                return reader.read(data_id)

        raise ConfigError('suitable data reader for {} not found'.format(data_id))
