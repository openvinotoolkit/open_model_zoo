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

import numpy as np

from ..config import BoolField, BaseField, NumberField, ConfigError, StringField
from ..preprocessor import Preprocessor
from ..utils import UnsupportedPackage
try:
    import scipy.signal as dsp
except ImportError as import_error:
    mask_util = UnsupportedPackage('scipy', import_error.msg)


class ResampleAudio(Preprocessor):
    __provider__ = 'resample_audio'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'sample_rate': NumberField(value_type=int, min_value=1,
                                       description="Set new audio sample rate."),
        })
        return parameters

    def configure(self):
        self.sample_rate = self.get_value_from_config('sample_rate')

    def process(self, image, annotation_meta=None):
        sample_rate = image.metadata.get('sample_rate')
        if sample_rate is None:
            raise RuntimeError('Operation "{}" can\'t resample audio: required original samplerate in metadata.'.
                               format(self.__provider__))

        if sample_rate == self.sample_rate:
            return image

        data = image.data
        duration = data.shape[1] / sample_rate
        resampled_data = np.zeros(shape=(data.shape[0], int(duration*self.sample_rate)), dtype=float)
        x_old = np.linspace(0, duration, data.shape[1])
        x_new = np.linspace(0, duration, resampled_data.shape[1])
        resampled_data[0] = np.interp(x_new, x_old, data[0])

        image.data = resampled_data
        image.metadata['sample_rate'] = self.sample_rate

        return image


class ClipAudio(Preprocessor):
    __provider__ = 'clip_audio'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'duration': BaseField(description="Length of audio clip in seconds or samples (with 'samples' suffix)."),
            'max_clips': NumberField(
                value_type=int, min_value=1, optional=True,
                description="Maximum number of clips per audiofile."
            ),
            'pad_to': NumberField(optional=True, default=0, description="Number of points each clip padded to."),
            'splice_frames': NumberField(optional=True, default=1, description="Number of frames to splice."),
            'pad_center': BoolField(optional=True, default=False, description="Place clip to center of padded frame"),
            'multi_infer': BoolField(optional=True, default=True, description="Metadata multi infer value"),
            'overlap': BaseField(optional=True, description="Overlapping part for each clip."),
        })
        return parameters

    def configure(self):
        duration = self.get_value_from_config('duration')
        self._parse_duration(duration)

        self.max_clips = self.get_value_from_config('max_clips') or np.inf

        overlap = self.get_value_from_config('overlap')
        self._parse_overlap(overlap)

        self.pad_to = int(self.get_value_from_config('pad_to'))
        self.pad_center = self.get_value_from_config('pad_center')
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.splice_frames = self.get_value_from_config('splice_frames')

    def process(self, image, annotation_meta=None):
        data = image.data

        sample_rate = image.metadata.get('sample_rate')
        if sample_rate is None:
            raise RuntimeError('Operation "{}" failed: required "sample rate" in metadata.'.
                               format(self.__provider__))
        audio_duration = data.shape[1]
        clip_duration = self.duration if self.is_duration_in_samples else int(self.duration * sample_rate)
        clipped_data = []
        if self.is_overlap_in_samples:
            hop = clip_duration - self.overlap_in_samples
        else:
            hop = int((1 - self.overlap) * clip_duration)

        if hop > clip_duration:
            raise ConfigError("Preprocessor {}: clip overlapping exceeds clip length.".format(self.__provider__))

        audio_duration = (audio_duration + hop - 1) // hop
        audio_duration = (audio_duration + self.splice_frames - 1) // self.splice_frames
        audio_duration *= self.splice_frames * hop
        audio_duration = int(audio_duration + clip_duration - hop)

        if audio_duration > data.shape[1]:
            data = np.concatenate((data, np.zeros((data.shape[0], audio_duration - data.shape[1]))), axis=1)

        for clip_no, clip_start in enumerate(range(0, audio_duration, hop)):
            if clip_start + clip_duration > audio_duration or clip_no >= self.max_clips:
                break
            clip = data[:, clip_start: clip_start + clip_duration]
            clipped_data.append(clip)

        if self.pad_to is not None:
            if self.pad_center:
                clipped_data = self._pad_center(np.asarray(clipped_data), self.pad_to)
        image.data = clipped_data
        image.metadata['multi_infer'] = self.multi_infer

        return image

    @staticmethod
    def _pad_center(data, size, axis=-1):
        n = data.shape[axis]
        lpad = int((size - n) // 2)
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (lpad, int(size - n - lpad))
        return np.pad(data, lengths)

    def _parse_overlap(self, overlap):
        self.is_overlap_in_samples = False
        self.overlap = 0
        if overlap is None:
            return
        if isinstance(overlap, str):
            if overlap.endswith('%'):
                try:
                    self.overlap = float(overlap[:-1]) / 100
                except ValueError:
                    raise ConfigError("Preprocessor {}: invalid value for 'overlap' - {}."
                                      .format(self.__provider__, overlap))
            elif overlap.endswith('samples'):
                try:
                    self.overlap_in_samples = int(overlap[:-7])
                except ValueError:
                    raise ConfigError("Preprocessor {}: invalid value for 'overlap' - {}."
                                      .format(self.__provider__, overlap))
                if self.overlap_in_samples < 1:
                    raise ConfigError("Preprocessor {}: invalid value for 'overlap' - {}."
                                      .format(self.__provider__, overlap))
                self.is_overlap_in_samples = True
            else:
                raise ConfigError("Preprocessor {}: invalid value for 'overlap' - {}."
                                  .format(self.__provider__, overlap))
        else:
            try:
                self.overlap = float(overlap)
            except ValueError:
                raise ConfigError("Preprocessor {}: invalid value for 'overlap' - {}."
                                  .format(self.__provider__, overlap))
            if self.overlap <= 0 or self.overlap >= 1:
                raise ConfigError("Preprocessor {}: invalid value for 'overlap' - {}."
                                  .format(self.__provider__, overlap))

    def _parse_duration(self, duration):
        self.is_duration_in_samples = False
        if isinstance(duration, str):
            if duration.endswith('samples'):
                try:
                    self.duration = int(duration[:-7])
                except ValueError:
                    raise ConfigError("Preprocessor {}: invalid value for duration - {}."
                                      .format(self.__provider__, duration))
                if self.duration <= 1:
                    raise ConfigError("Preprocessor {}: duration should be positive value - {}."
                                      .format(self.__provider__, self.duration))
                self.is_duration_in_samples = True
            else:
                raise ConfigError("Preprocessor {}: invalid value for duration - {}.".
                                  format(self.__provider__, duration))
        else:
            try:
                self.duration = float(duration)
            except ValueError:
                raise ConfigError("Preprocessor {}: invalid value for duration - {}."
                                  .format(self.__provider__, duration))
            if self.duration <= 0:
                raise ConfigError("Preprocessor {}: duration should be positive value - {}."
                                  .format(self.__provider__, self.duration))


class SamplesToFloat32(Preprocessor):
    __provider__ = 'audio_samples_to_float32'

    def process(self, image, annotation_meta=None):
        image.data = self._convert_samples_to_float32(image.data)
        return image

    @staticmethod
    def _convert_samples_to_float32(samples):
        """Convert sample type to float32.
        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= 1.0 / 2 ** (bits - 1)
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: {}.".format(samples.dtype))
        return float32_samples


class NormalizeAudio(Preprocessor):
    __provider__ = 'audio_normalization'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'int16mode': BoolField(optional=True, default=False, description="Normalization to int16 range"),
        })
        return parameters

    def configure(self):

        self.int16mode = self.get_value_from_config('int16mode')

    def process(self, image, annotation_meta=None):
        sound = image.data
        if self.int16mode:
            sound = sound / np.float32(0x8000)
        else:
            sound = (sound - np.mean(sound)) / (np.std(sound) + 1e-15)

        image.data = sound

        return image


class HanningWindow(Preprocessor):
    __provider__ = 'hanning_window'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'base': NumberField(
                optional=True,
                default=512,
                description="Window length",
                value_type=int
            ),
        })
        return parameters

    def configure(self):
        self.base = self.get_value_from_config('base')
        self.window = np.hanning(self.base)

    def process(self, image, annotation_meta=None):
        image.data = image.data * self.window

        return image


class AudioSpectrogram(Preprocessor):
    __provider__ = 'audio_spectrogram'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'fftbase': NumberField(optional=True, default=512, description="Base of FFT, samples", value_type=int),
            'magnitude_squared': BoolField(optional=True, default=True, description="Square spectrum magnitudes"),
            'skip_channels': BoolField(optional=True, default=False, description="Skips channels dimension"),
        })
        return parameters

    def configure(self):

        self.fftbase = self.get_value_from_config('fftbase')
        self.magnutide_squared = self.get_value_from_config('magnitude_squared')
        self.skip_channels = self.get_value_from_config('skip_channels')

    def process(self, image, annotation_meta=None):
        frames = image.data
        if self.skip_channels:
            frames = frames.squeeze()

        pspec = np.absolute(np.fft.rfft(frames, self.fftbase)) # pylint:disable=W9904
        if self.magnutide_squared:
            pspec = np.square(pspec)

        image.data = pspec

        return image


class TriangleFiltering(Preprocessor):
    __provider__ = 'audio_triangle_filtering'

    HZ2MEL = 1127.0
    MEL_CUTOFF = 700

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'base': NumberField(
                default=16000, description="Spectrogram length expected by filter bank", value_type=int
            ),
            'sample_rate': NumberField(
                default=16000, description="Sample rate value expected by model", value_type=int
            ),
            'filterbank_channel_count': NumberField(
                default=40, description="Number of channels in filter bank", value_type=int
            ),
            'lower_frequency_limit': NumberField(default=20, description="Filter passband lower boundary"),
            'upper_frequency_limit': NumberField(default=4000, description="Filter passband upper boundary"),
            'filter_amplitudes': BoolField(
                optional=True, default=False,
                description="Filter amplitude values (sqrt(power)) instead of power (Re^2+Im^2)"
            ),
        })
        return parameters

    def configure(self):
        self.base = self.get_value_from_config('base')
        self.sample_rate = self.get_value_from_config('sample_rate')
        self.filterbank_channel_count = self.get_value_from_config('filterbank_channel_count')
        self.lower_frequency_limit = self.get_value_from_config('lower_frequency_limit')
        self.upper_frequency_limit = self.get_value_from_config('upper_frequency_limit')
        self.filter_amplitudes = self.get_value_from_config('filter_amplitudes')
        self.initialize()

    def process(self, image, annotation_meta=None):
        spectrogram = image.data
        samples, _ = spectrogram.shape
        kFilterbankFloor = 1e-12

        mfcc_output = np.zeros((samples, self.filterbank_channel_count))

        for j in range(samples):
            filtered = self.compute(spectrogram[j, ...])
            filtered = np.where(filtered < kFilterbankFloor, kFilterbankFloor, filtered)
            mfcc_output[j, :] = np.log(filtered)

        image.data = mfcc_output

        return image

    def freq2mel(self, freq):
        return self.HZ2MEL * np.log1p(freq / self.MEL_CUTOFF)

    def initialize(self):
        center_frequencies = np.zeros((self.filterbank_channel_count + 1))
        mel_low = self.freq2mel(self.lower_frequency_limit)
        mel_hi = self.freq2mel(self.upper_frequency_limit)
        mel_span = mel_hi - mel_low
        mel_sapcing = mel_span / (self.filterbank_channel_count + 1)

        for i in range((self.filterbank_channel_count + 1)):
            center_frequencies[i] = mel_low + (mel_sapcing * (1 + i))

        hz_per_sbin = 0.5 * self.sample_rate / (self.base - 1)
        self.start_index = int(1.5 + (self.lower_frequency_limit / hz_per_sbin))
        self.end_index = int(self.upper_frequency_limit / hz_per_sbin)

        self.band_mapper = np.zeros(self.base)
        channel = 0

        for i in range(self.base):
            melf = self.freq2mel(i * hz_per_sbin)

            if ((i < self.start_index) or (i > self.end_index)):
                self.band_mapper[i] = -2
            else:
                while ((center_frequencies[int(channel)] < melf) and
                       (channel < self.filterbank_channel_count)):
                    channel += 1
                self.band_mapper[i] = channel - 1

        self.weights = np.zeros(self.base)
        for i in range(self.base):
            channel = self.band_mapper[i]
            if ((i < self.start_index) or (i > self.end_index)):
                self.weights[i] = 0.0
            else:
                if channel >= 0:
                    self.weights[i] = ((center_frequencies[int(channel) + 1] - self.freq2mel(i * hz_per_sbin))/
                                       (center_frequencies[int(channel) + 1] - center_frequencies[int(channel)]))
                else:
                    self.weights[i] = ((center_frequencies[0] - self.freq2mel(i * hz_per_sbin)) /
                                       (center_frequencies[0] - mel_low))

    def compute(self, mfcc_input):
        output_channels = np.zeros(self.filterbank_channel_count)
        for i in range(self.start_index, (self.end_index + 1)):
            if not self.filter_amplitudes:
                spec_val = mfcc_input[i]
            else:
                spec_val = np.sqrt(mfcc_input[i])
            weighted = spec_val * self.weights[i]
            channel = self.band_mapper[i]
            if channel >= 0:
                output_channels[int(channel)] += weighted
            channel += 1
            if channel < self.filterbank_channel_count:
                output_channels[int(channel)] += (spec_val - weighted)

        return output_channels


class DCT(Preprocessor):
    __provider__ = 'audio_dct'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'filterbank_channel_count': NumberField(
                default=40, description="Number of channels in filter bank", value_type=int
            ),
            'numceps': NumberField(default=26, description="Number of cepstral coefficients", value_type=int),
        })
        return parameters

    def configure(self):
        self.input_length = self.get_value_from_config('filterbank_channel_count')
        self.dct_coefficient_count = self.get_value_from_config('numceps')
        self.initialize()

    def process(self, image, annotation_meta=None):
        filtered = image.data

        samples, _ = filtered.shape
        cepstrum = np.zeros((samples, self.dct_coefficient_count))

        for j in range(samples):
            cepstrum[j, ...] = self.compute(filtered[j, ...])

        image.data = cepstrum

        return image

    def initialize(self):

        self.cosine = np.zeros((self.dct_coefficient_count, self.input_length))
        fnorm = np.sqrt(2.0 / self.input_length)
        arg = np.pi / self.input_length
        for i in range(self.dct_coefficient_count):
            for j in range(self.input_length):
                self.cosine[i][j] = fnorm * np.cos(i * arg * (j + 0.5))

    def compute(self, filtered):

        output_dct = np.zeros(self.dct_coefficient_count)
        base = filtered.shape[0]

        if base > self.input_length:
            base = self.input_length

        for i in range(self.dct_coefficient_count):
            _sum = 0.0
            for j in range(base):
                _sum += self.cosine[i][j] * filtered[j]
            output_dct[i] = _sum

        return output_dct


class ClipCepstrum(Preprocessor):
    __provider__ = 'clip_cepstrum'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'context': NumberField(default=9, description="Number of samples in context window", value_type=int),
            'numceps': NumberField(default=26, description="Number of input coefficients", value_type=int),
        })
        return parameters

    def configure(self):
        self.n_context = self.get_value_from_config('context')
        self.numceps = self.get_value_from_config('numceps')

    def process(self, image, annotation_meta=None):
        mfcc_feat = image.data

        num_strides, _ = mfcc_feat.shape
        empty_context = np.zeros((self.n_context, self.numceps), dtype=mfcc_feat.dtype)
        mfcc_feat = np.concatenate((empty_context, mfcc_feat, empty_context))

        window_size = 2 * self.n_context + 1
        features = np.lib.stride_tricks.as_strided(
            mfcc_feat,
            (num_strides, window_size, self.numceps),
            (mfcc_feat.strides[0], mfcc_feat.strides[0], mfcc_feat.strides[1]),
            writeable=False)

        image.data = features

        return image


class PackCepstrum(Preprocessor):
    __provider__ = 'pack_cepstrum'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'step': NumberField(default=16, description="Number of simultaneously processed contexts", value_type=int),
        })
        return parameters

    def configure(self):
        self.step = self.get_value_from_config('step')

    def process(self, image, annotation_meta=None):
        features = image.data

        steps, context, numceps = features.shape
        if steps % self.step:
            empty_context = np.zeros((self.step - (steps % self.step), context, numceps), dtype=features.dtype)
            features = np.concatenate((features, empty_context))
            steps, context, numceps = features.shape # pylint:disable=E0633

        packed = []
        for i in range(0, steps, self.step):
            packed.append(features[i:i+self.step, ...])

        image.data = packed
        image.metadata['multi_infer'] = True

        return image

class AddBatch(Preprocessor):
    __provider__ = 'add_batch'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'axis': NumberField(default=0, description="Add batch dimension", value_type=int),
        })
        return parameters

    def configure(self):
        self.axis = self.get_value_from_config('axis')

    def process(self, image, annotation_meta=None):
        data = image.data

        data = np.expand_dims(data, 0)
        if self.axis != 0:
            if self.axis > len(data.shape):
                raise RuntimeError(
                    'Operation "{}" failed: Invalid axis {} for shape {}.'.format(self.__provider__, self.axis,
                                                                                  data.shape)
                )
            order = list(range(1, self.axis + 1)) + [0] + list(range(self.axis + 1, len(data.shape)))
            data = np.transpose(data, order)
        image.data = data

        return image

class TrimmingAudio(Preprocessor):
    __provider__ = 'trim'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'top_db': NumberField(value_type=float, optional=True, default=60),
            'frame_length': NumberField(value_type=int, optional=True, default=2048, min_value=1),
            'hop_length': NumberField(value_type=int, optional=True, default=512, min_value=1)
        })
        return params

    def configure(self):
        self.top_db = self.get_value_from_config('top_db')
        self.frame_length = self.get_value_from_config('frame_length')
        self.hop_length = self.get_value_from_config('hop_length')

    def process(self, image, annotation_meta=None):
        image.data = self.trim(image.data)
        return image

    def trim(self, y):
        def frames_to_samples(frames, hop):
            return np.floor(np.asanyarray(frames) / hop).astype(int)

        non_silent = self._signal_to_frame_nonsilent(y)
        nonzero = np.flatnonzero(non_silent)

        if nonzero.size > 0:
            start = int(frames_to_samples(nonzero[0], self.hop_length))
            end = min(y.shape[-1], int(frames_to_samples(nonzero[-1] + 1, self.hop_length)))
        else:
            # The signal only contains zeros
            start, end = 0, 0

        # Build the mono/stereo index
        full_index = [slice(None)] * y.ndim
        full_index[-1] = slice(start, end)

        return y[tuple(full_index)]

    def _signal_to_frame_nonsilent(self, y):
        y_mono = np.asfortranarray(y)

        if y_mono.ndim > 1:
            y_mono = np.mean(y-y_mono, axis=0)

        # Compute the MSE for the signal
        mse = self.mse(y_mono)
        amin = 1e-10
        magnitude = mse.squeeze()
        ref_value = np.max(magnitude)
        log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
        log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

        return log_spec > (-1 * self.top_db)

    def mse(self, y_mono):
        y_mono = np.pad(y_mono, int(self.frame_length // 2), mode='reflect')
        n_frames = 1 + (y_mono.shape[-1] - self.frame_length) // self.hop_length
        strides = np.asarray(y_mono.strides)
        new_stride = np.prod(strides[strides > 0] // y_mono.itemsize) * y_mono.itemsize
        shape = list(y_mono.shape)[:-1] + [self.frame_length, n_frames]
        strides = list(strides) + [self.hop_length * new_stride]
        array = np.lib.stride_tricks.as_strided(y_mono, shape, strides)
        return np.mean(np.abs(array) ** 2, axis=0, keepdims=True)


def as_strided(x, shape, strides):
    class DummyArray:
        """Dummy object that just exists to hang __array_interface__ dictionaries
        and possibly keep alive a reference to a base array.
        """

        def __init__(self, interface, base=None):
            self.__array_interface__ = interface
            self.base = base

    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)

    array = np.asarray(DummyArray(interface, base=x))
    # The route via `__interface__` does not preserve structured
    # dtypes. Since dtype should remain unchanged, we set it explicitly.
    array.dtype = x.dtype
    return array

windows = {
    'hann': np.hanning,
    'hamming': np.hamming,
    'blackman': np.blackman,
    'bartlett': np.bartlett,
    'none': None,
}


class AudioToMelSpectrogram(Preprocessor):
    __provider__ = 'audio_to_mel_spectrogram'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'window_size': NumberField(optional=True, value_type=float, default=0.02,
                                       description="Size of frame in time-domain, seconds"),
            'window_stride': NumberField(optional=True, value_type=float, default=0.01,
                                         description="Intersection of frames in time-domain, seconds"),
            'window': StringField(
                choices=windows.keys(), optional=True, default='hann', description="Weighting window type"
            ),
            'n_fft': NumberField(optional=True, value_type=int, description="FFT base"),
            'n_filt': NumberField(optional=True, value_type=int, default=80, description="Number of MEL filters"),
            'splicing': NumberField(optional=True, value_type=int, default=1,
                                    description="Number of sequentially concastenated MEL spectrums"),
            'sample_rate': NumberField(optional=True, value_type=float, description="Audio samplimg frequency, Hz"),
            'pad_to': NumberField(optional=True, value_type=int, default=0, description="Desired length of features"),
            'preemph': NumberField(optional=True, value_type=float, default=0.97, description="Preemph factor"),
            'log': BoolField(optional=True, default=True, description="Enables log() of MEL features values"),
            'use_deterministic_dithering': BoolField(optional=True, default=True,
                                                     description="Applies determined dithering to signal spectrum"),
            'dither': NumberField(optional=True, value_type=float, default=0.00001, description="Dithering value"),
        })
        return params

    def configure(self):
        self.window_size = self.get_value_from_config('window_size')
        self.window_stride = self.get_value_from_config('window_stride')
        self.n_fft = self.get_value_from_config('n_fft')
        self.window_fn = windows.get(self.get_value_from_config('window'))
        self.preemph = self.get_value_from_config('preemph')
        self.nfilt = self.get_value_from_config('n_filt')
        self.sample_rate = self.get_value_from_config('sample_rate')
        self.log = self.get_value_from_config('log')
        self.pad_to = self.get_value_from_config('pad_to')
        self.frame_splicing = self.get_value_from_config('splicing')
        self.use_deterministic_dithering = self.get_value_from_config('use_deterministic_dithering')
        self.dither = self.get_value_from_config('dither')

        self.normalize = 'per_feature'
        self.lowfreq = 0
        self.highfreq = None
        self.max_duration = 16.7
        self.pad_value = 0
        self.mag_power = 2.0
        self.use_deterministic_dithering = True
        self.dither = 1e-05
        self.log_zero_guard_type = 'add'
        self.log_zero_guard_value = 2 ** -24

    def process(self, image, annotation_meta=None):
        if self.sample_rate is None:
            sample_rate = image.metadata.get('sample_rate')
            if sample_rate is None:
                raise RuntimeError(
                    'Operation "{}" failed: required "sample rate" in metadata.'.format(self.__provider__)
                )
        else:
            sample_rate = self.sample_rate
        self.window_length = int(self.window_size * sample_rate)
        self.hop_length = int(self.window_stride * sample_rate)
        self.n_fft = int(self.n_fft or 2 ** np.ceil(np.log2(self.window_length)))
        self.window = self.window_fn(self.window_length) if self.window_fn is not None else None
        highfreq = self.highfreq or (sample_rate / 2)
        filterbanks = np.expand_dims(self.mel(
            sample_rate, self.n_fft, n_mels=self.nfilt, fmin=self.lowfreq, fmax=highfreq
        ), 0)
        x = image.data
        seq_len = x.shape[-1]

        # Calculate maximum sequence length
        max_length = np.ceil(self.max_duration * sample_rate / self.hop_length)
        max_pad = self.pad_to - (max_length % self.pad_to) if self.pad_to > 0 else 0
        self.max_length = max_length + max_pad

        seq_len = int(np.ceil(seq_len // self.hop_length))

        # dither
        if self.dither > 0 and not self.use_deterministic_dithering:
            x = x + self.dither * np.random.randn(*x.shape)

        # do preemphasis
        if self.preemph is not None:
            x = np.concatenate((np.expand_dims(x[:, 0], 1), x[:, 1:] - self.preemph * x[:, :-1]), axis=1, )

        # do stft with weighting window
        _, _, x = dsp.stft(x.squeeze(), fs=sample_rate, window=self.window, nperseg=self.window_length,
                           noverlap=self.hop_length, nfft=self.n_fft)
        x *= sum(self.window)

        # get power spectrum
        if self.mag_power != 1.0:
            x = np.abs(x)**self.mag_power

        # dot with filterbank energies
        x = np.matmul(filterbanks, x)

        # log features if required
        if self.log:
            x = np.log(x + self.log_zero_guard_value)

        # frame splicing if required
        if self.frame_splicing > 1:
            x = self.splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            seq_len = x.shape[-1]
            x = self.normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of
        # `pad_to` (for efficiency)
        if self.pad_to:
            max_len = x.shape[-1]
            mask = np.arange(max_len)
            mask = mask >= seq_len
            x[:, :, mask] = self.pad_value
            del mask
            pad_to = self.pad_to
            if pad_to > 0:
                pad_amt = x.shape[-1] % pad_to
                if pad_amt != 0:
                    x = np.pad(x, ((0, 0), (0, 0), (0, pad_to - pad_amt)), constant_values=self.pad_value,
                               mode='constant')

        # transpose according to model input layout
        x = np.transpose(x, [2, 0, 1])

        image.data = x
        return image

    def mel(self, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, dtype=np.float32):
        if fmax is None:
            fmax = float(sr) / 2
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)
        # Center freqs of each FFT bin
        fftfreqs = np.linspace(0, float(sr) / 2, int(1 + n_fft // 2), endpoint=True)
        # 'Center freqs' of mel bands - uniformly spaced between limits
        mel_f = self.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)
        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)
        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

        return weights

    @staticmethod
    def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
        def hz_to_mel(frequencies):
            frequencies = np.asanyarray(frequencies)
            f_min = 0.0
            f_sp = 200.0 / 3
            mels = (frequencies - f_min) / f_sp
            min_log_hz = 1000.0
            min_log_mel = (min_log_hz - f_min) / f_sp
            logstep = np.log(6.4) / 27.0
            if frequencies.ndim:
                log_t = (frequencies >= min_log_hz)
                mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
            elif frequencies >= min_log_hz:
                mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
            return mels

        def mel_to_hz(mels):
            mels = np.asanyarray(mels)
            f_min = 0.0
            f_sp = 200.0 / 3
            freqs = f_min + f_sp * mels
            min_log_hz = 1000.0
            min_log_mel = (min_log_hz - f_min) / f_sp
            logstep = np.log(6.4) / 27.0

            if mels.ndim:
                log_t = (mels >= min_log_mel)
                freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
            elif mels >= min_log_mel:
                freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
            return freqs

        min_mel = hz_to_mel(fmin)
        max_mel = hz_to_mel(fmax)
        mels = np.linspace(min_mel, max_mel, n_mels)
        return mel_to_hz(mels)

    @staticmethod
    def stft(y, n_fft, hop_length, window, center=True, dtype=np.complex64, pad_mode='reflect'):
        # Constrain STFT block sizes to 256 KB
        MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10

        def pad_center(data, size, axis=-1, **kwargs):
            kwargs.setdefault('mode', 'constant')
            n = data.shape[axis]
            lpad = int((size - n) // 2)
            lengths = [(0, 0)] * data.ndim
            lengths[axis] = (lpad, int(size - n - lpad))
            return np.pad(data, lengths, **kwargs)

        def frame(x, frame_length=2048, hop_length=512):
            """Slice a data array into (overlapping) frames."""

            n_frames = (x.shape[-1] - frame_length) // hop_length
            strides = np.asarray(x.strides)

            new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize
            shape = list(x.shape)[:-1] + [frame_length, n_frames]
            strides = list(strides) + [hop_length * new_stride]

            return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        fft_window = np.asarray(window)

        # Pad the window out to n_fft size
        fft_window = pad_center(fft_window, n_fft)

        # Reshape so that the window can be broadcast
        fft_window = fft_window.reshape((-1, 1))
        # # Pad the time series so that frames are centered
        if center:
            y = np.pad(y, int(n_fft // 2), mode=pad_mode)

        # Window the time series.
        y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

        # Pre-allocate the STFT matrix
        stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                               dtype=dtype,
                               order='F')

        # how many columns can we fit within MAX_MEM_BLOCK?
        n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                         stft_matrix.itemsize))

        for bl_s in range(0, stft_matrix.shape[1], n_columns):
            bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

            # RFFT and Conjugate here to match phase from DPWE code
            rfft = fft_window * y_frames[:, bl_s:bl_t]
            stft_matrix[:, bl_s:bl_t] = np.fft.rfft(rfft, axis=0)[:stft_matrix.shape[0]]

        return stft_matrix

    @staticmethod
    def splice_frames(x, frame_splicing):
        seq = [x]
        for n in range(1, frame_splicing):
            tmp = np.zeros_like(x)
            tmp[:, :, :-n] = x[:, :, n:]
            seq.append(tmp)
        return np.concatenate(seq, axis=1)[:, :, ::frame_splicing]

    @staticmethod
    def normalize_batch(x, seq_len, normalize_type):
        if normalize_type == 'per_feature':
            x_mean = np.zeros((x.shape[0], x.shape[1]), dtype=x.dtype)
            x_std = np.zeros((x.shape[0], x.shape[1]), dtype=x.dtype)
            for i in range(x.shape[0]):
                x_mean[i, :] = x[i, :, :seq_len].mean(axis=1)
                x_std[i, :] = x[i, :, :seq_len].std(axis=1)
            # make sure x_std is not zero
            x_std += 1e-5
            return (x - np.expand_dims(x_mean, 2)) / np.expand_dims(x_std, 2)

        if normalize_type == 'all_features':
            x_mean = np.zeros(seq_len, dtype=x.dtype)
            x_std = np.zeros(seq_len, dtype=x.dtype)
            for i in range(x.shape[0]):
                x_mean[i] = x[i, :, : seq_len[i].item()].mean()
                x_std[i] = x[i, :, : seq_len[i].item()].std()
            # make sure x_std is not zero
            x_std += 1e-5
            return x - np.reshape(x_mean, (-1, 1, 1)) / np.reshape(x_std, (-1, 1, 1))
        return x
