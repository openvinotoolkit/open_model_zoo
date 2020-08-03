"""
Copyright (c) 2018-2020 Intel Corporation

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

from pathlib import Path
from functools import singledispatch
from collections import OrderedDict, namedtuple
import re
import wave

import cv2
from PIL import Image
import numpy as np
from numpy.lib.npyio import NpzFile

try:
    import tensorflow as tf
except ImportError as import_error:
    tf = None

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import pydicom
except ImportError:
    pydicom = None


from ..utils import get_path, read_json, contains_all
from ..dependency import ClassProvider
from ..config import BaseField, StringField, ConfigValidator, ConfigError, DictField, ListField, BoolField

REQUIRES_ANNOTATIONS = ['annotation_features_extractor', ]


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


def create_reader(config):
    return BaseReader.provide(config.get('type', 'opencv_imread'), config.get('data_source'), config=config)


class DataReaderField(BaseField):
    def validate(self, entry_, field_uri=None):
        super().validate(entry_, field_uri)

        if entry_ is None:
            return

        field_uri = field_uri or self.field_uri
        if isinstance(entry_, str):
            StringField(choices=BaseReader.providers).validate(entry_, 'reader')
        elif isinstance(entry_, dict):
            class DictReaderValidator(ConfigValidator):
                type = StringField(choices=BaseReader.providers)

            dict_reader_validator = DictReaderValidator(
                'reader', on_extra_argument=DictReaderValidator.IGNORE_ON_EXTRA_ARGUMENT
            )
            dict_reader_validator.validate(entry_)
        else:
            self.raise_error(entry_, field_uri, 'reader must be either string or dictionary')


class BaseReader(ClassProvider):
    __provider_type__ = 'reader'

    def __init__(self, data_source, config=None, **kwargs):
        self.config = config or {}
        self.data_source = data_source
        self.read_dispatcher = singledispatch(self.read)
        self.read_dispatcher.register(list, self._read_list)
        self.read_dispatcher.register(ClipIdentifier, self._read_clip)
        self.read_dispatcher.register(MultiFramesInputIdentifier, self._read_frames_multi_input)
        self.multi_infer = False

        self.validate_config()
        self.configure()

    def __call__(self, identifier):
        return self.read_item(identifier)

    def configure(self):
        if not self.data_source:
            raise ConfigError('data_source parameter is required to create "{}" '
                              'data reader and read data'.format(self.__provider__))
        self.data_source = get_path(self.data_source, is_directory=True)
        self.multi_infer = self.config.get('multi_infer', False)

    def validate_config(self):
        pass

    def read(self, data_id):
        raise NotImplementedError

    def _read_list(self, data_id):
        return [self.read(identifier) for identifier in data_id]

    def _read_clip(self, data_id):
        video = Path(data_id.video)
        frames_identifiers = [video / frame for frame in data_id.frames]
        return self.read_dispatcher(frames_identifiers)

    def _read_frames_multi_input(self, data_id):
        return self.read_dispatcher(data_id.frames)

    def read_item(self, data_id):
        data_rep = DataRepresentation(self.read_dispatcher(data_id), identifier=data_id)
        if self.multi_infer:
            data_rep.metadata['multi_infer'] = True
        return data_rep

    @property
    def name(self):
        return self.__provider__

    def reset(self):
        pass


class ReaderCombinerConfig(ConfigValidator):
    type = StringField()
    scheme = DictField(
        value_type=DataReaderField(), key_type=StringField(), allow_empty=False
    )


class ReaderCombiner(BaseReader):
    __provider__ = 'combine_reader'

    def validate_config(self):
        config_validator = ReaderCombinerConfig('reader_combiner_config')
        config_validator.validate(self.config)

    def configure(self):
        scheme = self.config['scheme']
        reading_scheme = OrderedDict()
        for pattern, reader_config in scheme.items():
            reader_type = reader_config['type'] if isinstance(reader_config, dict) else reader_config
            reader_configuration = reader_config if isinstance(reader_config, dict) else None
            reader = BaseReader.provide(reader_type, self.data_source, reader_configuration)
            pattern = re.compile(pattern)
            reading_scheme[pattern] = reader

        self.reading_scheme = reading_scheme
        self.multi_infer = self.config.get('multi_infer', False)

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


class OpenCVImageReaderConfig(ConfigValidator):
    type = StringField(optional=True)
    reading_flag = StringField(optional=True, choices=OPENCV_IMREAD_FLAGS, default='color')


class OpenCVImageReader(BaseReader):
    __provider__ = 'opencv_imread'

    def validate_config(self):
        if self.config:
            config_validator = OpenCVImageReaderConfig(
                'opencv_imread_config', on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT
            )
            config_validator.validate(self.config)

    def configure(self):
        super().configure()
        self.flag = OPENCV_IMREAD_FLAGS[self.config.get('reading_flag', 'color') if self.config else 'color']

    def read(self, data_id):
        return cv2.imread(str(get_path(self.data_source / data_id)), self.flag)


class PillowImageReader(BaseReader):
    __provider__ = 'pillow_imread'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config)
        self.convert_to_rgb = True

    def read(self, data_id):
        with open(str(self.data_source / data_id), 'rb') as f:
            img = Image.open(f)

            return np.array(img.convert('RGB') if self.convert_to_rgb else img)


class ScipyImageReader(BaseReader):
    __provider__ = 'scipy_imread'

    def read(self, data_id):
        # reimplementation scipy.misc.imread
        image = Image.open(str(get_path(self.data_source / data_id)))
        if image.mode == 'P':
            image = image.convert('RGBA') if 'transparency' in image.info else image.convert('RGB')

        return np.array(image)


class OpenCVFrameReader(BaseReader):
    __provider__ = 'opencv_capture'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config)
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
                raise EOFError('frame with {} index does not exists in {}'.format(self.current, self.data_source))

        return frame

    def configure(self):
        if not self.data_source:
            raise ConfigError('data_source parameter is required to create "{}" '
                              'data reader and read data'.format(self.__provider__))
        self.data_source = get_path(self.data_source)
        self.videocap = cv2.VideoCapture(str(self.data_source))
        self.multi_infer = self.config.get('multi_infer', False)

    def reset(self):
        self.current = -1
        self.videocap.set(cv2.CAP_PROP_POS_FRAMES, 0)


class JSONReaderConfig(ConfigValidator):
    type = StringField()
    key = StringField(optional=True, case_sensitive=True)


class JSONReader(BaseReader):
    __provider__ = 'json_reader'

    def validate_config(self):
        config_validator = JSONReaderConfig('json_reader_config')
        config_validator.validate(self.config)

    def configure(self):
        self.key = self.config.get('key')
        self.multi_infer = self.config.get('multi_infer', False)
        if not self.data_source:
            raise ConfigError('data_source parameter is required to create "{}" '
                              'data reader and read data'.format(self.__provider__))

    def read(self, data_id):
        data = read_json(str(self.data_source / data_id))
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


class NiftyReaderConfig(ConfigValidator):
    type = StringField(optional=True)
    channels_first = BoolField(optional=True, default=False)


class NiftiImageReader(BaseReader):
    __provider__ = 'nifti_reader'

    def validate_config(self):
        if self.config:
            config_validator = NiftyReaderConfig('nifti_reader_config')
            config_validator.validate(self.config)

    def configure(self):
        if nib is None:
            raise ImportError('nifty backend for image reading requires nibabel. Please install it before usage.')
        self.channels_first = self.config.get('channels_first', False) if self.config else False
        self.multi_infer = self.config.get('multi_infer', False)
        if not self.data_source:
            raise ConfigError('data_source parameter is required to create "{}" '
                              'data reader and read data'.format(self.__provider__))

    def read(self, data_id):
        nib_image = nib.load(str(get_path(self.data_source / data_id)))
        image = np.array(nib_image.dataobj)
        if len(image.shape) != 4:  # Make sure 4D
            image = np.expand_dims(image, -1)
        image = np.transpose(image, (3, 0, 1, 2) if self.channels_first else (2, 1, 0, 3))

        return image


class NumpyReaderConfig(ConfigValidator):
    type = StringField(optional=True)
    keys = StringField(optional=True, default="")
    separator = StringField(optional=True, default="@")


class NumPyReader(BaseReader):
    __provider__ = 'numpy_reader'

    def validate_config(self):
        if self.config:
            config_validator = NumpyReaderConfig('numpy_reader_config')
            config_validator.validate(self.config)

    def configure(self):
        self.is_text = self.config.get('text_file', False)
        self.multi_infer = self.config.get('multi_infer', False)
        self.keys = self.config.get('keys', "") if self.config else ""
        self.keys = [t.strip() for t in self.keys.split(',')] if len(self.keys) > 0 else []
        self.separator = self.config.get('separator')
        if self.separator and self.is_text:
            raise ConfigError('text file reading with numpy does')
        self.multi_infer = self.config.get('multi_infer', False)
        if not self.data_source:
            raise ConfigError('data_source parameter is required to create "{}" '
                              'data reader and read data'.format(self.__provider__))
        self.keyRegex = {k: re.compile(k) for k in self.keys}
        self.valRegex = re.compile(r"([^0-9]+)([0-9]+)")

    def read(self, data_id):
        field_id = None
        if self.separator:
            field_id, data_id = data_id.split(self.separator)

        data = np.load(str(self.data_source / data_id))

        if not isinstance(data, NpzFile):
            return data

        if field_id is not None:
            key = [k for k, v in self.keyRegex.items() if v.match(field_id)]
            if len(key) > 0:
                recno = field_id.split('_')[-1]
                recno = int(recno)
                start, _ = Path(data_id).name.split('.')
                start = int(start)
                return data[key[0]][recno - start, :]

        key = next(iter(data.keys()))
        return data[key]


class NumpyTXTReader(BaseReader):
    __provider__ = 'numpy_txt_reader'

    def read(self, data_id):
        return np.loadtxt(str(self.data_source / data_id))


class TensorflowImageReader(BaseReader):
    __provider__ = 'tf_imread'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config)
        if tf is None:
            raise ImportError('tf backend for image reading requires TensorFlow. Please install it before usage.')

        tf.enable_eager_execution()

        def read_func(path):
            img_raw = tf.read_file(str(path))
            img_tensor = tf.image.decode_image(img_raw, channels=3)
            return img_tensor.numpy()

        self.read_realisation = read_func

    def read(self, data_id):
        return self.read_realisation(self.data_source / data_id)


class AnnotationFeaturesConfig(ConfigValidator):
    type = StringField()
    features = ListField(allow_empty=False, value_type=StringField)


class AnnotationFeaturesReader(BaseReader):
    __provider__ = 'annotation_features_extractor'

    def configure(self):
        self.feature_list = self.config['features']
        if not contains_all(self.data_source[0].__dict__, self.feature_list):
            raise ConfigError(
                'annotation_class prototype does not contain provided features {}'.format(', '.join(self.feature_list))
            )
        self.single = len(self.feature_list) == 1
        self.counter = 0
        self.subset = range(len(self.data_source))
        self.multi_infer = self.config.get('multi_infer', False)

    def read(self, data_id):
        relevant_annotation = self.data_source[self.subset[self.counter]]
        self.counter += 1
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

    def read(self, data_id):
        with wave.open(str(self.data_source / data_id), "rb") as wav:
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            nframes = wav.getnframes()
            data = wav.readframes(nframes)
            if self._samplewidth_types.get(sample_width):
                data = np.frombuffer(data, dtype=self._samplewidth_types[sample_width])
            else:
                raise RuntimeError("Reader {} coudn't process file {}: unsupported sample width {}"
                                   "(reader only supports {})"
                                   .format(self.__provider__, str(self.data_source / data_id),
                                           sample_width, [*self._samplewidth_types.keys()]))
            data = data.reshape(-1, wav.getnchannels()).T

        return data, {'sample_rate': sample_rate}

    def read_item(self, data_id):
        return DataRepresentation(*self.read_dispatcher(data_id), identifier=data_id)


class DicomReader(BaseReader):
    __provider__ = 'dicom_reader'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config)
        if pydicom is None:
            raise ImportError('dicom backend for reading requires pydicom. Please install it before usage.')

    def read(self, data_id):
        dataset = pydicom.dcmread(str(self.data_source / data_id))
        return dataset.pixel_array
