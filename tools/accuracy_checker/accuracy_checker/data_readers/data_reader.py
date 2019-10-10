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
try:
    import tensorflow as tf
except ImportError as import_error:
    tf = None

# For audio:
import scipy.io.wavfile as wav

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
        self.config = config
        self.data_source = data_source
        self.read_dispatcher = singledispatch(self.read)
        self.read_dispatcher.register(list, self._read_list)
        self.read_dispatcher.register(ClipIdentifier, self._read_clip)
        self.read_dispatcher.register(MultiFramesInputIdentifier, self._read_frames_multi_input)

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

    def _read_frames_multi_input(self, data_id):
        return self.read_dispatcher(data_id.frames)

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


class AudioReader(BaseReader):
    __provider__ = 'audio_reader'

    ##################################################
    #
    # John Feng, 2019/09/06
    #
    ##################################################

    @staticmethod
    def audio_spectrogram(samples, window_size, stride, magnitude_squared):
        # window_size, #=(16000 * (32 / 1000)), #Config.audio_window_samples,
        # stride, # =(16000 * (20 / 1000)), #(Config.audio_step_samples,
        # magnitude_squared) : # =True) :
        if len(samples.shape) != 2:
            raise ConfigError("input must be 2-dimensional")

        window_size = int(window_size)
        stride = int(stride)

        sample_count = samples.shape[0]
        channel_count = samples.shape[1] # == 1

        def output_frequency_channels(n):
            _log = np.floor(np.log2(n))
            _log = _log if ((n == (n & ~(n - 1)))) else _log + 1

            fft_length = 1 << _log.astype(np.int32)

            return 1 + fft_length / 2, fft_length.astype(np.int32)

        output_width, fft_length = output_frequency_channels(window_size)
        output_width = output_width.astype(np.int32)

        length_minus_windows = sample_count - window_size

        output_height = 0 if length_minus_windows < 0 else (1 + (length_minus_windows / stride))
        output_height = int(output_height)
        output_slices = channel_count

        __output = np.zeros((output_slices, output_height, output_width))
        hann_window = np.hanning(window_size)

        for i in range(channel_count):
            input_for_channel = samples[:, i]

            input_for_compute = np.zeros(stride)
            spectrogram = np.zeros((output_height, output_width))

            fft_input_output = np.zeros(fft_length)

            for j in range(output_height):
                start = j * stride
                end = start + window_size
                if end < sample_count:
                    input_for_compute = input_for_channel[start: end]

                    fft_input_output[0 :window_size] = input_for_compute * hann_window
                    fft_input_output[window_size:] = 0

                    _f = np.fft.rfft(fft_input_output.astype(np.float32), n=fft_length)

                    spectrogram[j] = np.real(_f) ** 2 +  np.imag(_f) ** 2

            __output = spectrogram if (magnitude_squared) else np.sqrt(spectrogram)

            return __output


    ##################################################
    #
    # John Feng, 2019/09/11
    #
    ##################################################

    @staticmethod
    def mfcc_mel_filiterbank_init(sample_rate, input_length):
        # init
        filterbank_channel_count_ = 40
        lower_frequency_limit_ = 20
        upper_frequency_limit_ = 4000

        def freq2mel(freq):
            return 1127.0 * np.log1p(freq / 700)

        center_frequencies = np.zeros((filterbank_channel_count_ + 1))
        mel_low = freq2mel(lower_frequency_limit_)
        mel_hi = freq2mel(upper_frequency_limit_)
        mel_span = mel_hi - mel_low
        mel_sapcing = mel_span / (filterbank_channel_count_ + 1)

        for i in range((filterbank_channel_count_ + 1)):
            center_frequencies[i] = mel_low + (mel_sapcing * (1 + i))

        hz_per_sbin = 0.5 * sample_rate / (input_length - 1)
        start_index = int(1.5 + (lower_frequency_limit_ / hz_per_sbin))
        end_index = int(upper_frequency_limit_ / hz_per_sbin)

        band_mapper = np.ones(input_length) * -2
        channel = 0

        for i in range(input_length):
            melf = freq2mel(i * hz_per_sbin)
            if start_index <= i <= end_index:
                while ((center_frequencies[int(channel)] < melf) and
                       (channel < filterbank_channel_count_)):
                    channel += 1
                band_mapper[i] = channel - 1

        weights = np.zeros(input_length, dtype=np.float32)

        for i in range(input_length):
            channel = band_mapper[i]
            if start_index <= i <= end_index:
                if channel >= 0:
                    weights[i] = ((center_frequencies[int(channel) + 1] - freq2mel(i * hz_per_sbin)) /
                                  (center_frequencies[int(channel) + 1] - center_frequencies[int(channel)]))
                else:
                    weights[i] = ((center_frequencies[0] - freq2mel(i * hz_per_sbin)) /
                                  (center_frequencies[0] - mel_low))

        return start_index, end_index, weights, band_mapper

    @staticmethod
    def mfcc_mel_filiterbank_compute(mfcc_input, input_length, start_index, end_index, weights, band_mapper):
        filterbank_channel_count_ = 40
        # Compute
        output_channels = np.zeros(filterbank_channel_count_)
        for i in range(start_index, (end_index + 1)):
            spec_val = np.sqrt(mfcc_input[i])
            weighted = spec_val * weights[i]
            channel = band_mapper[i]
            if channel >= 0:
                output_channels[int(channel)] += weighted
            channel += 1
            if channel < filterbank_channel_count_:
                output_channels[int(channel)] += (spec_val - weighted)

        return output_channels

    @staticmethod
    def dct_init(input_length, dct_coefficient_count):
        # init
        if input_length < dct_coefficient_count:
            raise ConfigError("Error input_length need to larger than dct_coefficient_count")

        cosine = np.zeros((dct_coefficient_count, input_length))
        fnorm = np.sqrt(2.0 / input_length)
        arg = np.pi / input_length
        for i in range(dct_coefficient_count):
            for j in range(input_length):
                cosine[i][j] = fnorm * np.cos(i * arg * (j + 0.5))

        return cosine

    @staticmethod
    def dct_compute(worked_filiter, input_length, dct_coefficient_count, cosine):
        # compute
        output_dct = np.zeros(dct_coefficient_count)
        worked_length = worked_filiter.shape[0]

        if worked_length > input_length:
            worked_length = input_length

        for i in range(dct_coefficient_count):
            _sum = 0.0
            for j in range(worked_length):
                _sum += cosine[i][j] * worked_filiter[j]
            output_dct[i] = _sum

        return output_dct

    @staticmethod
    def mfcc(spectrogram, sample_rate, dct_coefficient_count):
        audio_channels, spectrogram_samples, spectrogram_channels = spectrogram.shape
        kFilterbankFloor = 1e-12
        filterbank_channel_count = 40

        mfcc_output = np.zeros((spectrogram_samples, dct_coefficient_count))
        for i in range(audio_channels):
            start_index, end_index, weights, band_mapper = self.mfcc_mel_filiterbank_init(sample_rate,
                                                                                          spectrogram_channels)
            cosine = self.dct_init(filterbank_channel_count, dct_coefficient_count)
            for j in range(spectrogram_samples):
                mfcc_input = spectrogram[i, j, :]

                mel_filiter = self.mfcc_mel_filiterbank_compute(mfcc_input, spectrogram_channels,
                                                                start_index, end_index,
                                                                weights, band_mapper)
                for k in range(mel_filiter.shape[0]):
                    val = mel_filiter[k]
                    if val < kFilterbankFloor:
                        val = kFilterbankFloor

                    mel_filiter[k] = np.log(val)

                mfcc_output[j, :] = self.dct_compute(mel_filiter,
                                                     filterbank_channel_count,
                                                     dct_coefficient_count,
                                                     cosine)

        return mfcc_output

    def read(self, data_id):
        fs, audio = wav.read(str(get_path(self.data_source / data_id)))

        audio = audio/np.float32(32768) # normalize to -1 to 1, int 16 to float32
        audio = audio.reshape(-1, 1)
        spectrogram = self.audio_spectrogram(audio, (16000 * 32 / 1000), (16000 * 20 / 1000), True)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        features = self.mfcc(spectrogram, fs, 26)

        return features


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
