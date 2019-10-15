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

######################################
#
#   John Feng, john.feng@intel.com
#
######################################


from ..config import NumberField
from .preprocessor import Preprocessor

class CreateOverlapWindows(Preprocessor):
    __provider__ = 'overlap_creator'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'step': NumberField(
                value_type=int, optional=True, min_value=1, description="Number of step length."
            ),
            'context': NumberField(
                value_type=int, optional=True, min_value=1, description="Number of context length."
            ),
            'input': NumberField(
                value_type=int, optional=True, min_value=1, description="Number of input size."
            )
        })

        return parameters

    def configure(self):
        self.step = self.get_value_from_config('step')
        self.context = self.get_value_from_config('context')
        self.input = self.get_value_from_config('input')

    def process(self, image, annotation_meta=None):
        empty_context = np.zeros((self.context, self.input), dtype=image.data.dtype)
        audio = np.concatenate((empty_context, image.data, empty_context))

        num_strides = len(audio) - (self.context * 2)
        window_size = 2 * self.context + 1

        audio = np.lib.stride_tricks.as_strided(audio,
                                                (num_strides, window_size, self.input),
                                                (audio.strides[0], audio.strides[0], audio.strides[1]),
                                                writeable=False)
        image.data = audio

        return image

class PrepareAudioPackage(Preprocessor):
    __provider__ = 'audio_package'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'step': NumberField(
                value_type=int, optional=True, min_value=1, description="Number of step length."
            )
        })

        return parameters

    def configure(self):
        self.step = self.get_value_from_config('step')

    def process(self, image, annotation_meta=None):
        data = image.data

        _length, c, i = data.shape

        _p_pad = _length % self.step
        if _p_pad != 0:
            data = np.pad(data, ((0, self.step - _p_pad), (0, 0), (0, 0)), mode='constant', constant_values=0)

        image.data = np.transpose(data.reshape((-1, self.step, c, i)), (0, 2, 3, 1))

        return image

class AudioNormalizer(Preprocessor):
    __provider__ = 'audio_normalizer'

    def process(self, image, annotation_meta=None):
        image.data = image.data / np.float32(32767)

        return image

class AudioSpectrogram(Preprocessor):
    __provider__ = 'audio_spectrogram'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'rate': NumberField(
                value_type=int, optional=True, min_value=1, description="Sample rate"
            ),
            'window_length':NumberField(
                value_type=int, optional=True, min_value=1, description="audio window length for feature extration"
            ),
            'window_step':NumberField(
                value_type=int, optional=True, min_value=1, description="window step length for feature extration"
            )
        })

        return parameters

    def configure(self):
        self.rate = self.get_value_from_config('rate')
        self.window_length = self.get_value_from_config('window_length')
        self.window_step = self.get_value_from_config('window_step')

    @staticmethod
    def audio_spectrogram(samples, window_size, stride, magnitude_squared):
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

    def process(self, image, annotation_meta=None):
        image.data = self.audio_spectrogram(image.data,
                                            (self.rate * self.window_length / 1000),
                                            (self.rate * self.window_step / 1000),
                                            True)

        return image

class MfccFeature(Preprocessor):
    __provider__ = 'mfcc_feature'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'rate': NumberField(
                value_type=int, optional=True, min_value=1, description="Sample rate"
            )
        })

        return parameters

    def configure(self):
        self.rate = self.get_value_from_config('rate')

    @staticmethod
    def mfcc(spectrogram, sample_rate, dct_coefficient_count):
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

        audio_channels, spectrogram_samples, spectrogram_channels = spectrogram.shape
        kFilterbankFloor = 1e-12
        filterbank_channel_count = 40

        mfcc_output = np.zeros((spectrogram_samples, dct_coefficient_count))
        for i in range(audio_channels):
            start_index, end_index, weights, band_mapper = mfcc_mel_filiterbank_init(sample_rate,
                                                                                     spectrogram_channels)
            cosine = dct_init(filterbank_channel_count, dct_coefficient_count)
            for j in range(spectrogram_samples):
                mfcc_input = spectrogram[i, j, :]

                mel_filiter = mfcc_mel_filiterbank_compute(mfcc_input,
                                                           spectrogram_channels,
                                                           start_index,
                                                           end_index,
                                                           weights,
                                                           band_mapper)
                for k in range(mel_filiter.shape[0]):
                    val = mel_filiter[k]
                    if val < kFilterbankFloor:
                        val = kFilterbankFloor

                    mel_filiter[k] = np.log(val)

                mfcc_output[j, :] = dct_compute(mel_filiter,
                                                filterbank_channel_count,
                                                dct_coefficient_count,
                                                cosine)

        return mfcc_output

    def process(self, image, annotation_meta=None):
        spectrogram = np.expand_dims(image.data, axis=0)
        image.data = self.mfcc(spectrogram, self.rate, 26)

        return image