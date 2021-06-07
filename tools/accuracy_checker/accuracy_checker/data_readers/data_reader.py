"""
Copyright (c) 2018-2021 Intel Corporation

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

import struct
import re
import wave
from collections import OrderedDict, namedtuple
from functools import singledispatch
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.npyio import NpzFile
from PIL import Image

from ..utils import (
    get_path, read_json, read_pickle, contains_all, contains_any, UnsupportedPackage, get_parameter_value_from_config
)
from ..dependency import ClassProvider, UnregisteredProviderException
from ..config import (
    BaseField, StringField, ConfigValidator, ConfigError, DictField, ListField, BoolField, NumberField, PathField
)

try:
    import lmdb
except ImportError as import_error:
    lmdb = UnsupportedPackage("lmdb", import_error.msg)

try:
    import nibabel as nib
except ImportError as import_error:
    nib = UnsupportedPackage("nibabel", import_error.msg)

try:
    import pydicom
except ImportError as import_error:
    pydicom = UnsupportedPackage("pydicom", import_error.msg)

try:
    import skimage.io as sk
except ImportError as import_error:
    sk = UnsupportedPackage('skimage.io', import_error.msg)

try:
    import rawpy
except ImportError as import_error:
    rawpy = UnsupportedPackage('rawpy', import_error.msg)

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


ClipIdentifier = namedtuple('ClipIdentifier', ['video', 'clip_id', 'frames'])
MultiFramesInputIdentifier = namedtuple('MultiFramesInputIdentifier', ['input_id', 'frames'])
ImagePairIdentifier = namedtuple('ImagePairIdentifier', ['first', 'second'])
ListIdentifier = namedtuple('ListIdentifier', ['values'])
MultiInstanceIdentifier = namedtuple('MultiInstanceIdentifier', ['identifier', 'object_id'])
KaldiMatrixIdentifier = namedtuple('KaldiMatrixIdentifier', ['file', 'key'])
KaldiFrameIdentifier = namedtuple('KaldiFrameIdentifier', ['file', 'key', 'id'])
ParametricImageIdentifier = namedtuple('ParametricImageIdentifier', ['identifier', 'parameters'])

IdentifierSerializationOptions = namedtuple(
    "identifierSerializationOptions", ['type', 'fields', 'class_id', 'recursive', 'to_tuple']
)

identifier_serialization = {
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
    )
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


def create_identifier_key(identifier):
    if isinstance(identifier, list):
        return ListIdentifier(tuple(identifier))
    if isinstance(identifier, ClipIdentifier):
        return ClipIdentifier(identifier.video, identifier.clip_id, tuple(identifier.frames))
    if isinstance(identifier, MultiFramesInputIdentifier):
        return MultiFramesInputIdentifier(tuple(identifier.input_id), tuple(identifier.frames))
    if isinstance(identifier, ParametricImageIdentifier):
        return ParametricImageIdentifier(identifier.identifier, tuple(identifier.parameters))
    return identifier


def create_reader(config):
    return BaseReader.provide(config.get('type', 'opencv_imread'), config.get('data_source'), config=config)


class DataReaderField(BaseField):
    def validate(self, entry_, field_uri=None, fetch_only=False, validation_scheme=None):
        errors = super().validate(entry_, field_uri)

        if entry_ is None:
            return errors

        field_uri = field_uri or self.field_uri
        if isinstance(entry_, str):
            errors.extend(
                StringField(choices=BaseReader.providers).validate(
                    entry_, field_uri, fetch_only=fetch_only, validation_scheme=validation_scheme)
            )
        elif isinstance(entry_, dict):
            class DictReaderValidator(ConfigValidator):
                type = StringField(choices=BaseReader.providers)

            dict_reader_validator = DictReaderValidator(
                'reader', on_extra_argument=DictReaderValidator.IGNORE_ON_EXTRA_ARGUMENT
            )
            errors.extend(
                dict_reader_validator.validate(
                    entry_, field_uri, fetch_only=fetch_only, validation_scheme=validation_scheme
                ))
        else:
            msg = 'reader must be either string or dictionary'
            if not fetch_only:
                self.raise_error(entry_, field_uri, msg)
            errors.append(self.build_error(entry_, field_uri, msg, validation_scheme))

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

    def read_item(self, data_id):
        data_rep = DataRepresentation(
            self.read_dispatcher(data_id),
            identifier=data_id if not isinstance(data_id, ListIdentifier) else list(data_id.values)
        )
        if self.multi_infer:
            data_rep.metadata['multi_infer'] = True
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

    def read(self, data_id):
        for pattern, reader in self.reading_scheme.items():
            if pattern.match(str(data_id)):
                return reader.read(data_id)

        raise ConfigError('suitable data reader for {} not found'.format(data_id))


OPENCV_IMREAD_FLAGS = {
    'color': cv2.IMREAD_COLOR,
    'gray': cv2.IMREAD_GRAYSCALE,
    'unchanged': cv2.IMREAD_UNCHANGED
}


class OpenCVImageReader(BaseReader):
    __provider__ = 'opencv_imread'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'reading_flag': StringField(optional=True, choices=OPENCV_IMREAD_FLAGS, default='color',
                                        description='Flag which specifies the way image should be read.')
        })
        return parameters

    def configure(self):
        super().configure()
        self.flag = OPENCV_IMREAD_FLAGS[self.get_value_from_config('reading_flag')]

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source else data_id
        return cv2.imread(str(get_path(data_path)), self.flag)


class PillowImageReader(BaseReader):
    __provider__ = 'pillow_imread'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config, **kwargs)
        self.convert_to_rgb = True

    def read(self, data_id):
        data_path = get_path(self.data_source / data_id) if self.data_source is not None else data_id
        with open(str(data_path), 'rb') as f:
            img = Image.open(f)

            return np.array(img.convert('RGB') if self.convert_to_rgb else img)


class ScipyImageReader(BaseReader):
    __provider__ = 'scipy_imread'

    def read(self, data_id):
        # reimplementation scipy.misc.imread
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        image = Image.open(str(get_path(data_path)))
        if image.mode == 'P':
            image = image.convert('RGBA') if 'transparency' in image.info else image.convert('RGB')

        return np.array(image)


class OpenCVFrameReader(BaseReader):
    __provider__ = 'opencv_capture'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config, **kwargs)
        self.current = -1

    def read(self, data_id):
        if data_id < 0:
            raise IndexError('frame with {} index can not be grabbed, non-negative index is expected')
        if data_id < self.current:
            self.videocap.set(cv2.CAP_PROP_POS_FRAMES, data_id)
            self.current = data_id - 1

        return self._read_sequence(data_id)

    def _read_sequence(self, data_id):
        frame = None
        while self.current != data_id:
            success, frame = self.videocap.read()
            self.current += 1
            if not success:
                raise EOFError('frame with {} index does not exist in {}'.format(self.current, self.data_source))

        return frame

    def configure(self):
        if not self.data_source:
            raise ConfigError('data_source parameter is required to create "{}" '
                              'data reader and read data'.format(self.__provider__))
        self.data_source = get_path(self.data_source)
        self.videocap = cv2.VideoCapture(str(self.data_source))
        self.multi_infer = self.get_value_from_config('multi_infer')

    def reset(self):
        self.current = -1
        self.videocap.set(cv2.CAP_PROP_POS_FRAMES, 0)


class JSONReader(BaseReader):
    __provider__ = 'json_reader'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'key': StringField(optional=True, case_sensitive=True,
                               description='Key for reading from json dictionary.')
        })
        return parameters

    def configure(self):
        self.key = self.get_value_from_config('key')
        self.multi_infer = self.get_value_from_config('multi_infer')
        if not self.data_source:
            if not self._postpone_data_source:
                raise ConfigError('data_source parameter is required to create "{}" '
                                  'data reader and read data'.format(self.__provider__))
        else:
            self.data_source = get_path(self.data_source, is_directory=True)

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        data = read_json(data_path)
        if self.key:
            data = data.get(self.key)

            if not data:
                raise ConfigError('{} does not contain {}'.format(data_id, self.key))

        return np.array(data).astype(np.float32)


class NCFDataReader(BaseReader):
    __provider__ = 'ncf_data_reader'

    def configure(self):
        pass

    def read(self, data_id):
        if not isinstance(data_id, str):
            raise IndexError('Data identifier must be a string')

        return float(data_id.split(":")[1])


class NiftiImageReader(BaseReader):
    __provider__ = 'nifti_reader'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'channels_first': BoolField(optional=True, default=False,
                                        description='Allows read files and transpose in order where channels first.'),
            'frame_separator': StringField(optional=True, default='#',
                                           description="Separator between filename and frame number"),
            'multi_frame': BoolField(optional=True, default=False,
                                     description="Add annotation for each frame in source file"),
            'to_4D': BoolField(optional=True, default=True, description="Ensure that data are 4D"),
            'frame_axis': NumberField(optional=True, default=-1, description="Frames dimension axis"),
        })
        return parameters

    def configure(self):
        if isinstance(nib, UnsupportedPackage):
            nib.raise_error(self.__provider__)
        self.channels_first = self.get_value_from_config('channels_first')
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.frame_axis = int(self.get_value_from_config('frame_axis'))
        self.frame_separator = self.get_value_from_config('frame_separator')
        self.multi_frame = self.get_value_from_config('multi_frame')
        self.to_4D = self.get_value_from_config('to_4D')

        if not self.data_source:
            if not self._postpone_data_source:
                raise ConfigError('data_source parameter is required to create "{}" '
                                  'data reader and read data'.format(self.__provider__))
        else:
            self.data_source = get_path(self.data_source, is_directory=True)

    def read(self, data_id):
        if self.multi_frame:
            parts = data_id.split(self.frame_separator)
            frame_number = int(parts[1])
            data_id = parts[0]
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        nib_image = nib.load(str(get_path(data_path)))
        image = np.array(nib_image.dataobj)
        if self.multi_frame:
            image = image[:, :, frame_number]
            image = np.expand_dims(image, 0)
        if self.to_4D:
            if len(image.shape) != 4:  # Make sure 4D
                image = np.expand_dims(image, -1)
            image = np.transpose(image, (3, 0, 1, 2) if self.channels_first else (2, 1, 0, 3))

        return image


class NumPyReader(BaseReader):
    __provider__ = 'numpy_reader'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'keys': StringField(optional=True, default="", description='Comma-separated model input names.'),
            'separator': StringField(optional=True,
                                     description='Separator symbol between input identifier and file identifier.'),
            'id_sep': StringField(
                optional=True, default="_",
                description='Separator symbol between input name and record number in input identifier.'
            ),
            'block': BoolField(optional=True, default=False, description='Allows block mode.'),
            'batch': NumberField(optional=True, default=1, description='Batch size')
        })
        return parameters

    def configure(self):
        self.is_text = self.config.get('text_file', False)
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.keys = self.get_value_from_config('keys')
        self.keys = [t.strip() for t in self.keys.split(',')] if len(self.keys) > 0 else []
        self.separator = self.get_value_from_config('separator')
        self.id_sep = self.get_value_from_config('id_sep')
        self.block = self.get_value_from_config('block')
        self.batch = int(self.get_value_from_config('batch'))

        if self.separator and self.is_text:
            raise ConfigError('text file reading with numpy does')
        if not self.data_source:
            if not self._postpone_data_source:
                raise ConfigError('data_source parameter is required to create "{}" '
                                  'data reader and read data'.format(self.__provider__))
        else:
            self.data_source = get_path(self.data_source, is_directory=True)
        self.keyRegex = {k: re.compile(k + self.id_sep) for k in self.keys}
        self.valRegex = re.compile(r"([^0-9]+)([0-9]+)")

    def read(self, data_id):
        field_id = None
        if self.separator:
            field_id, data_id = str(data_id).split(self.separator)
        data_path = self.data_source / data_id if self.data_source is not None else data_id

        data = np.load(str(data_path))

        if not isinstance(data, NpzFile):
            return data

        if field_id is not None:
            key = [k for k, v in self.keyRegex.items() if v.match(field_id)]
            if len(key) > 0:
                if self.block:
                    res = data[key[0]]
                else:
                    recno = field_id.split('_')[-1]
                    recno = int(recno)
                    start = Path(data_id).name.split('.')[0]
                    start = int(start)
                    res = data[key[0]][recno - start, :]
                return res

        key = next(iter(data.keys()))
        return data[key]


class NumpyTXTReader(BaseReader):
    __provider__ = 'numpy_txt_reader'

    def read(self, data_id):
        return np.loadtxt(str(self.data_source / data_id))


class NumpyDictReader(BaseReader):
    __provider__ = 'numpy_dict_reader'

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        return np.load(str(data_path), allow_pickle=True)[()]

    def read_item(self, data_id):
        dict_data = self.read_dispatcher(data_id)
        identifier = []
        data = []
        for key, value in dict_data.items():
            identifier.append('{}.{}'.format(data_id, key))
            data.append(value)
        if len(data) == 1:
            return DataRepresentation(data[0], identifier=data_id)
        return DataRepresentation(data, identifier=identifier)


class NumpyBinReader(BaseReader):
    __provider__ = 'numpy_bin_reader'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            "dtype": StringField(optional=True, default='float32', description='data type for reading')
        })
        return params

    def configure(self):
        super().configure()
        self.dtype = self.get_value_from_config('dtype')

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        return np.fromfile(data_path, dtype=self.dtype)


class TensorflowImageReader(BaseReader):
    __provider__ = 'tf_imread'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config, **kwargs)
        try:
            import tensorflow as tf  # pylint: disable=C0415
        except ImportError as import_error:
            raise ImportError(
                'tf backend for image reading requires TensorFlow. '
                'Please install it before usage. {}'.format(import_error.msg)
            )
        if tf.__version__ < '2.0.0':
            tf.enable_eager_execution()

        def read_func(path):
            img_raw = tf.read_file(str(path)) if tf.__version__ < '2.0.0' else tf.io.read_file(str(path))
            img_tensor = tf.image.decode_image(img_raw, channels=3)
            return img_tensor.numpy()

        self.read_realisation = read_func

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        return self.read_realisation(data_path)


class AnnotationFeaturesReader(BaseReader):
    __provider__ = 'annotation_features_extractor'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'features': ListField(allow_empty=False, value_type=str, description='List of features.')
        })
        return parameters

    def configure(self):
        self.feature_list = self.get_value_from_config('features')
        self.single = len(self.feature_list) == 1
        self.multi_infer = self.get_value_from_config('multi_infer')

    def read(self, data_id):
        relevant_annotation = self.data_source[create_identifier_key(data_id)]
        if not contains_all(relevant_annotation.__dict__, self.feature_list):
            raise ConfigError(
                'annotation_class prototype does not contain provided features {}'.format(', '.join(self.feature_list))
            )
        features = [getattr(relevant_annotation, feature) for feature in self.feature_list]
        if self.single:
            return features[0]
        return features

    def _read_list(self, data_id):
        return self.read(data_id)

    def reset(self):
        self.subset = range(len(self.data_source))
        self.counter = 0


class WavReader(BaseReader):
    __provider__ = 'wav_reader'

    _samplewidth_types = {
        1: np.uint8,
        2: np.int16
    }

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'mono': BoolField(optional=True, default=False,
                              description='get mean along channels if multichannel audio loaded'),
            'to_float': BoolField(optional=True, default=False, description='converts audio signal to float')
        })
        return params

    def configure(self):
        super().configure()
        self.mono = self.get_value_from_config('mono')
        self.to_float = self.get_value_from_config('to_float')

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        with wave.open(str(data_path), "rb") as wav:
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            nframes = wav.getnframes()
            data = wav.readframes(nframes)
            if self._samplewidth_types.get(sample_width):
                data = np.frombuffer(data, dtype=self._samplewidth_types[sample_width])
            else:
                raise RuntimeError("Reader {} couldn't process file {}: unsupported sample width {}"
                                   "(reader only supports {})"
                                   .format(self.__provider__, self.data_source / data_id,
                                           sample_width, [*self._samplewidth_types.keys()]))
            channels = wav.getnchannels()

            data = data.reshape(-1, channels).T
            if channels > 1 and self.mono:
                data = data.mean(0, keepdims=True)
            if self.to_float:
                data = data.astype(np.float32) / np.iinfo(self._samplewidth_types[sample_width]).max

        return data, {'sample_rate': sample_rate}

    def read_item(self, data_id):
        return DataRepresentation(*self.read_dispatcher(data_id), identifier=data_id)


class DicomReader(BaseReader):
    __provider__ = 'dicom_reader'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config)
        if isinstance(pydicom, UnsupportedPackage):
            pydicom.raise_error(self.__provider__)

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        dataset = pydicom.dcmread(str(data_path))
        return dataset.pixel_array


class PickleReader(BaseReader):
    __provider__ = 'pickle_reader'

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        data = read_pickle(data_path)
        if isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict):
            return data

        return data, {}

    def read_item(self, data_id):
        return DataRepresentation(*self.read_dispatcher(data_id), identifier=data_id)


class SkimageReader(BaseReader):
    __provider__ = 'skimage_imread'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config, **kwargs)
        if isinstance(sk, UnsupportedPackage):
            sk.raise_error(self.__provider__)

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        return sk.imread(str(data_path))


class RawpyReader(BaseReader):
    __provider__ = 'rawpy'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'postprocess': BoolField(optional=True, default=True)
        })
        return params

    def configure(self):
        if isinstance(rawpy, UnsupportedPackage):
            rawpy.raise_error(self.__provider__)
        self.postprocess = self.get_value_from_config('postprocess')
        super().configure()

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        raw = rawpy.imread(str(data_path))
        if not self.postprocess:
            return raw.raw_image_visible.astype(np.float32)
        postprocessed = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        return np.float32(postprocessed / 65535.0)


class ByteFileReader(BaseReader):
    __provider__ = 'byte_reader'

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        with open(data_path, 'rb') as f:
            return np.array(f.read())


class LMDBReader(BaseReader):
    __provider__ = 'lmdb_reader'

    def configure(self):
        super().configure()
        self.database = lmdb.open(bytes(self.data_source), readonly=True)

    def read(self, data_id):
        with self.database.begin(write=False) as txn:
            img_key = f'image-{data_id:09d}'.encode()
            image_bytes = txn.get(img_key)
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            if len(img.shape) < 3:
                img = np.stack((img,) * 3, axis=-1)
            assert img.shape[-1] == 3
            return img


class KaldiARKReader(BaseReader):
    __provider__ = 'kaldi_ark_reader'

    def configure(self):
        super().configure()
        self.buffer = {}

    @staticmethod
    def read_frames(in_file):
        ut = {}
        with open(str(in_file), 'rb') as fd:
            while True:
                try:
                    key = KaldiARKReader.read_token(fd)
                    if not key:
                        break
                    binary = fd.read(2).decode()
                    if binary == ' [':
                        mat = KaldiARKReader.read_ascii_mat(fd)
                    else:
                        ark_type = KaldiARKReader.read_token(fd)
                        float_size = 4 if ark_type[0] == 'F' else 8
                        float_type = np.float32 if ark_type[0] == 'F' else float
                        num_rows = KaldiARKReader.read_int32(fd)
                        num_cols = KaldiARKReader.read_int32(fd)
                        mat_data = fd.read(float_size * num_cols * num_rows)
                        mat = np.frombuffer(mat_data, dtype=float_type).reshape(num_rows, num_cols)
                    ut[key] = mat
                except EOFError:
                    break
            return ut

    def read_utterance(self, file_name, utterance):
        if file_name not in self.buffer:
            self.buffer[file_name] = self.read_frames(self.data_source / file_name)
        return self.buffer[file_name][utterance]

    def read_frame(self, file_name, utterance, idx):
        return self.read_utterance(file_name, utterance)[idx]

    @staticmethod
    def read_int32(fd):
        int_size = bytes.decode(fd.read(1))
        assert int_size == '\04', 'Expect \'\\04\', but gets {}'.format(int_size)
        int_str = fd.read(4)
        int_val = struct.unpack('i', int_str)
        return int_val[0]

    @staticmethod
    def read_token(fd):
        key = ''
        while True:
            c = bytes.decode(fd.read(1))
            if c in [' ', '', '\0', '\4']:
                break
            key += c
        return None if key == '' else key.strip()

    def read(self, data_id, reset=True):
        assert (
            isinstance(data_id, (KaldiMatrixIdentifier, KaldiFrameIdentifier))
        ), "Kaldi reader support only Kaldi specific data IDs"
        file_id = data_id.file
        if file_id not in self.buffer and self.buffer and reset:
            self.reset()
        if len(data_id) == 3:
            return self.read_frame(data_id.file, data_id.key, data_id.id)
        matrix = self.read_utterance(data_id.file, data_id.key)
        if self.multi_infer:
            matrix = list(matrix)
        return matrix

    def _read_list(self, data_id):
        if not contains_any(self.buffer, [data.file for data in data_id]) and self.buffer:
            self.reset()

        return [self.read(idx, reset=False) for idx in data_id]

    def reset(self):
        del self.buffer
        self.buffer = {}

    @staticmethod
    def read_ascii_mat(fd):
        rows = []
        while True:
            line = fd.readline().decode()
            if not line.strip():
                continue # skip empty line
            arr = line.strip().split()
            if arr[-1] != ']':
                rows.append(np.array(arr, dtype='float32')) # not last line
            else:
                rows.append(np.array(arr[:-1], dtype='float32')) # last line
                mat = np.vstack(rows)
                return mat
