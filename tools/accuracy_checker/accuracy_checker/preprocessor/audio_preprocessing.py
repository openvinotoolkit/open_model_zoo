"""
Copyright (c) 2020 Intel Corporation

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

from ..config import BoolField, BaseField, NumberField, ConfigError
from ..preprocessor import Preprocessor


class ResampleAudio(Preprocessor):
    __provider__ = 'resample_audio'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'sample_rate': NumberField(value_type=int, min_value=1,
                                       description='Set new audio sample rate.'),
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
            'duration': BaseField(
                description="Length of audio clip in seconds or samples (with 'samples' suffix)."
            ),
            'max_clips': NumberField(
                value_type=int, min_value=1, optional=True,
                description="Maximum number of clips per audiofile."
            ),
            'overlap': BaseField(
                optional=True,
                description="Overlapping part for each clip."
            ),
        })
        return parameters

    def configure(self):
        duration = self.get_value_from_config('duration')
        self._parse_duration(duration)

        self.max_clips = self.get_value_from_config('max_clips') or np.inf

        overlap = self.get_value_from_config('overlap')
        self._parse_overlap(overlap)

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

        for clip_no, clip_start in enumerate(range(0, audio_duration, hop)):
            if clip_start + clip_duration > audio_duration or clip_no >= self.max_clips:
                break
            clip = data[:, clip_start: clip_start + clip_duration]
            clipped_data.append(clip)

        image.data = clipped_data
        image.metadata['multi_infer'] = True

        return image

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
            sound = sound / np.float32(0x7fff)
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
                description="Window length"
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
            'fftbase': NumberField(optional=True, default=512, description="Base of FFT, samples"),
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
            "base": NumberField(default=16000, description='Spectrogram length expected by filter bank'),
            "sample_rate": NumberField(default=16000, description='sample rate value expected by model'),
            "filterbank_channel_count": NumberField(default=40, description='number of channels in filter bank'),
            "lower_frequency_limit": NumberField(default=20, description='filter passband lower boundary'),
            "upper_frequency_limit": NumberField(default=4000, description='filter passband upper boundary'),
        })
        return parameters

    def configure(self):
        self.base = self.get_value_from_config('base')
        self.sample_rate = self.get_value_from_config('sample_rate')
        self.filterbank_channel_count = self.get_value_from_config('filterbank_channel_count')
        self.lower_frequency_limit = self.get_value_from_config('lower_frequency_limit')
        self.upper_frequency_limit = self.get_value_from_config('upper_frequency_limit')
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
            "filterbank_channel_count": NumberField(default=40, description='number of channels in filter bank'),
            "numceps": NumberField(default=26, description='Number of cepstral coefficients'),
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
            "context": NumberField(default=9, description='number of samples in context window'),
            "numceps": NumberField(default=26, description='Number of input coefficients'),
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
            "step": NumberField(default=16, description='number of simultaneously processed contexts'),
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

        features = np.expand_dims(features, 0)

        packed = []
        for i in range(0, steps, self.step):
            packed.append(features[:, i:i+self.step, ...])

        image.data = packed
        image.metadata['multi_infer'] = True

        return image
