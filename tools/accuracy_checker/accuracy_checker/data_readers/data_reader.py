"""
Copyright (c) 2019 Intel Corporation

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
import cv2
from PIL import Image
import numpy as np
import nibabel as nib

from ..utils import get_path, read_json, zipped_transform, set_image_metadata, contains_all
from ..dependency import ClassProvider
from ..config import BaseField, StringField, ConfigValidator, ConfigError, DictField, ListField


class DataRepresentation:
    def __init__(self, data, meta=None, identifier=''):
        self.identifier = identifier
        self.data = data
        self.metadata = meta or {}
        if np.isscalar(data):
            self.metadata['image_size'] = 1
        elif isinstance(data, list) and np.isscalar(data[0]):
            self.metadata['image_size'] = len(data)
        else:
            self.metadata['image_size'] = data.shape if not isinstance(data, list) else data[0].shape


ClipIdentifier = namedtuple('ClipIdentifier', ['video', 'clip_id', 'frames'])


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
        self.config = config
        self.data_source = data_source
        self.read_dispatcher = singledispatch(self.read)
        self.read_dispatcher.register(list, self._read_list)
        self.read_dispatcher.register(ClipIdentifier, self._read_clip)

        self.validate_config()
        self.configure()

    def __call__(self, context=None, identifier=None, **kwargs):
        if identifier is not None:
            return self.read_item(identifier)

        if not context:
            raise ValueError('identifier or context should be specified')

        read_data = [self.read_item(identifier) for identifier in context.identifiers_batch]
        context.data_batch = read_data
        context.annotation_batch, context.data_batch = zipped_transform(
            set_image_metadata,
            context.annotation_batch,
            context.data_batch
        )
        return context

    def configure(self):
        self.data_source = get_path(self.data_source, is_directory=True)

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

    def read_item(self, data_id):
        return DataRepresentation(self.read_dispatcher(data_id), identifier=data_id)


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
            reader = BaseReader.provide(
                reader_config['type'] if isinstance(reader_config, dict) else reader_config,
                self.data_source, reader_config
            )
            pattern = re.compile(pattern)
            reading_scheme[pattern] = reader

        self.reading_scheme = reading_scheme

    def read(self, data_id):
        for pattern, reader in self.reading_scheme.items():
            if pattern.match(str(data_id)):
                return reader.read(data_id)

        raise ConfigError('suitable data reader for {} not found'.format(data_id))


class OpenCVImageReader(BaseReader):
    __provider__ = 'opencv_imread'

    def read(self, data_id):
        return cv2.imread(str(get_path(self.data_source / data_id)))


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

    @staticmethod
    def _from_image(image, flatten=False, mode=None):
        if mode is not None:
            if mode != image.mode:
                image = image.convert(mode)
        elif image.mode == 'P':
            image = image.convert('RGBA') if 'transparency' in image.info else image.convert('RGB')

        if flatten:
            image = image.convert('F')
        elif image.mode == '1':
            image = image.convert('L')

        return np.array(image)

    @staticmethod
    def _to_image(arr, high=255, low=0, cmin=None, cmax=None, pal=None, mode=None, channel_axis=None):
        _errstr = "Mode is unknown or incompatible with input array shape."

        def process_2d(data, shape, pal, cmax, cmin):
            shape = (shape[1], shape[0])  # columns show up first
            if mode == 'F':
                data32 = data.astype(np.float32)
                image = Image.frombytes(mode, shape, data32.tostring())

                return image
            if mode in [None, 'L', 'P']:
                bytedata = ScipyImageReader._bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
                image = Image.frombytes('L', shape, bytedata.tostring())
                if pal is not None:
                    image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                    # Becomes a mode='P' automagically.
                elif mode == 'P':  # default gray-scale
                    pal1 = np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis]
                    pal2 = np.ones((3,), dtype=np.uint8)[np.newaxis, :]
                    pal = pal1 * pal2
                    image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())

                return image
            if mode == '1':  # high input gives threshold for 1
                bytedata = (data > high)
                image = Image.frombytes('1', shape, bytedata.tostring())

                return image

            cmin = cmin or np.amin(np.ravel(data))
            cmax = cmax or np.amax(np.ravel(data))
            data = (data * 1.0 - cmin) * (high - low) / (cmax - cmin) + low
            if mode == 'I':
                data32 = data.astype(np.uint32)
                image = Image.frombytes(mode, shape, data32.tostring())
            else:
                raise ValueError(_errstr)

            return image

        def process_3d(data, shape, mode):
            # if here then 3-d array with a 3 or a 4 in the shape length.
            # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
            if channel_axis is None:
                ca = np.flatnonzero(np.asarray(shape) == 3) if 3 in shape else np.flatnonzero(np.asarray(shape) == 4)
                if not np.size(ca):
                    raise ValueError("Could not find channel dimension.")
                ca = ca[0]
            else:
                ca = channel_axis

            numch = shape[ca]
            if numch not in [3, 4]:
                raise ValueError("Channel axis dimension is not valid.")

            bytedata = ScipyImageReader._bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
            channel_axis_mapping = {
                0: ((1, 2, 0), (shape[1], shape[0])),
                1: ((0, 2, 1), (shape[2], shape[0])),
                2: ((0, 1, 2), (shape[1], shape[0]))
            }
            if ca in channel_axis_mapping:
                transposition, shape = channel_axis_mapping[ca]
                strdata = np.transpose(bytedata, transposition).tostring()

            if mode is None:
                mode = 'RGB' if numch == 3 else 'RGBA'

            if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
                raise ValueError(_errstr)

            if mode in ['RGB', 'YCbCr'] and numch != 3:
                raise ValueError("Invalid array shape for mode.")
            if mode in ['RGBA', 'CMYK'] and numch != 4:
                raise ValueError("Invalid array shape for mode.")

            # Here we know data and mode is correct
            image = Image.frombytes(mode, shape, strdata)
            return image

        data = np.asarray(arr)
        if np.iscomplexobj(data):
            raise ValueError("Cannot convert a complex-valued array.")
        shape = list(data.shape)
        valid = len(shape) == 2 or ((len(shape) == 3) and ((3 in shape) or (4 in shape)))
        if not valid:
            raise ValueError("'arr' does not have a suitable array shape for any mode.")
        if len(shape) == 2:
            return process_2d(data, shape, pal, cmax, cmin)
        return process_3d(data, shape, mode)

    @staticmethod
    def _imread(name):
        # reimplementation scipy.misc.imread
        image = Image.open(name)

        return ScipyImageReader._from_image(image)

    @staticmethod
    def _bytescale(data, cmin=None, cmax=None, high=255, low=0):
        if data.dtype == np.uint8:
            return data

        if high > 255:
            raise ValueError("`high` should be less than or equal to 255.")
        if low < 0:
            raise ValueError("`low` should be greater than or equal to 0.")
        if high < low:
            raise ValueError("`high` should be greater than or equal to `low`.")
        cmin = cmin or data.min()
        cmax = cmax or data.max()

        cscale = cmax - cmin
        if cscale < 0:
            raise ValueError("`cmax` should be larger than `cmin`.")
        if cscale == 0:
            cscale = 1

        scale = float(high - low) / cscale
        bytedata = (data - cmin) * scale + low

        return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

    @staticmethod
    def imresize(arr, size, interp='bilinear', mode=None):
        im = ScipyImageReader._to_image(arr, mode=mode)
        ts = type(size)
        if np.issubdtype(ts, np.signedinteger):
            percent = size / 100.0
            size = tuple((np.array(im.size) * percent).astype(int))
        elif np.issubdtype(type(size), np.floating):
            size = tuple((np.array(im.size) * size).astype(int))
        else:
            size = (size[1], size[0])
        func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
        imnew = im.resize(size, resample=func[interp])

        return ScipyImageReader._from_image(imnew)

    def read(self, data_id):
        return ScipyImageReader._imread(str(get_path(self.data_source / data_id)))


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
        self.data_source = get_path(self.data_source)
        self.videocap = cv2.VideoCapture(str(self.data_source))


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


class NiftiImageReader(BaseReader):
    __provider__ = 'nifti_reader'

    def read(self, data_id):
        nib_image = nib.load(str(get_path(self.data_source / data_id)))
        image = np.array(nib_image.dataobj)
        if len(image.shape) != 4:  # Make sure 4D
            image = np.expand_dims(image, -1)
        image = np.swapaxes(np.array(image), 0, -2)

        return image


class NumPyReader(BaseReader):
    __provider__ = 'numpy_reader'

    def read(self, data_id):
        return np.load(str(self.data_source / data_id))


class TensorflowImageReader(BaseReader):
    __provider__ = 'tf_imread'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config)
        try:
            import tensorflow as tf
        except ImportError as import_error:
            raise ConfigError(
                'tf_imread reader disabled.Please, install Tensorflow before using. \n{}'.format(import_error.msg)
            )

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

    def __init__(self, data_source, config=None, annotations=None):
        super().__init__(annotations, config)
        self.counter = 0
        self.data_source = annotations

    def configure(self):
        self.feature_list = self.config['features']
        if not contains_all(self.data_source[0].__dict__, self.feature_list):
            raise ConfigError(
                'annotation_class prototype does not contain provided features {}'.format(', '.join(self.feature_list))
            )
        self.single = len(self.feature_list) == 1

    def read(self, data_id):
        relevant_annotation = self.data_source[self.counter]
        self.counter += 1
        features = [getattr(relevant_annotation, feature) for feature in self.feature_list]
        if self.single:
            return features[0]
        return features

    def _read_list(self, data_id):
        return self.read(data_id)
