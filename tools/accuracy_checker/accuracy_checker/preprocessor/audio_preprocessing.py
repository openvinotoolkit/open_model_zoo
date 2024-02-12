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

import numpy as np

from ..config import BoolField, NumberField, StringField
from ..preprocessor import Preprocessor
from ..utils import UnsupportedPackage

try:
    import scipy.signal as dsp
except ImportError as import_error:
    dsp = UnsupportedPackage('scipy', import_error.msg)


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
    shape_modificator = True

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
        pspec = np.absolute(np.fft.rfft(frames, self.fftbase))  # pylint:disable=bad-function-call
        if self.magnutide_squared:
            pspec = np.square(pspec)
        image.data = pspec
        return image

    def calculate_out_single_shape(self, data_shape):
        fake_input = np.zeros(data_shape)
        if self.skip_channels:
            fake_input = fake_input.squeeze()
        return np.fft.rfft(fake_input, self.fftbase).shape

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]


class FFTSpectrogram(Preprocessor):
    __provider__ = 'fft'
    shape_modificator = True

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'num_fft': NumberField(optional=True, default=512, description="Base of FFT, samples", value_type=int),
            'magnitude_squared': BoolField(optional=True, default=True, description="Square spectrum magnitudes"),
            'skip_channels': BoolField(optional=True, default=False, description="Skips channels dimension"),
        })
        return parameters

    def configure(self):
        self.num_fft = self.get_value_from_config('num_fft')
        self.magnutide_squared = self.get_value_from_config('magnitude_squared')
        self.skip_channels = self.get_value_from_config('skip_channels')

    def process(self, image, annotation_meta=None):
        frames = image.data
        if self.skip_channels:
            frames = frames.squeeze()
        pspec = np.abs(np.fft.fft(frames, n=self.num_fft))
        if self.magnutide_squared:
            pspec = np.square(pspec)
        image.data = pspec
        return image

    def calculate_out_single_shape(self, data_shape):
        fake_input = np.zeros(data_shape)
        if self.skip_channels:
            fake_input = fake_input.squeeze()
        return np.fft.fft(fake_input, n=self.num_fft).shape

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]


class TriangleFiltering(Preprocessor):
    __provider__ = 'audio_triangle_filtering'
    shape_modificator = True

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

    def calculate_out_single_shape(self, data_shape):
        samples, _ = data_shape
        return samples, self.filterbank_channel_count

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]

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
                    self.weights[i] = ((center_frequencies[int(channel) + 1] - self.freq2mel(i * hz_per_sbin)) /
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
    shape_modificator = True

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

    def calculate_out_single_shape(self, data_shape):
        if -1 in data_shape:
            return data_shape
        samples, _ = data_shape
        return samples, self.dct_coefficient_count

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]

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
    shape_modificator = True

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
        image.data = self.process_features(image.data)

        return image

    def process_features(self, mfcc_feat):
        num_strides, _ = mfcc_feat.shape
        empty_context = np.zeros((self.n_context, self.numceps), dtype=mfcc_feat.dtype)
        mfcc_feat = np.concatenate((empty_context, mfcc_feat, empty_context))

        window_size = 2 * self.n_context + 1
        features = np.lib.stride_tricks.as_strided(
            mfcc_feat,
            (num_strides, window_size, self.numceps),
            (mfcc_feat.strides[0], mfcc_feat.strides[0], mfcc_feat.strides[1]),
            writeable=False)
        return features

    def calculate_out_single_shape(self, data_shape):
        if -1 in data_shape:
            return data_shape
        data = np.zeros(data_shape)
        feats = self.process_features(data)
        return feats.shape

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]


class PackCepstrum(Preprocessor):
    __provider__ = 'pack_cepstrum'
    shape_modificator = True

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
        packed = self.process_features(features)

        image.data = packed
        image.metadata['multi_infer'] = True

        return image

    def process_features(self, features):
        steps, context, numceps = features.shape
        if steps % self.step:
            empty_context = np.zeros((self.step - (steps % self.step), context, numceps), dtype=features.dtype)
            features = np.concatenate((features, empty_context))
            steps, context, numceps = features.shape  # pylint:disable=E0633

        packed = []
        for i in range(0, steps, self.step):
            packed.append(features[i:i + self.step, ...])
        return packed

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]

    def calculate_out_single_shape(self, data_shape):
        data = np.zeros(data_shape)
        packed = self.process_features(data)
        return np.shape(packed)


class TrimmingAudio(Preprocessor):
    __provider__ = 'trim'
    shape_modificator = True

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
            y_mono = np.mean(y - y_mono, axis=0)

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

    @staticmethod
    def calculate_out_shape(data_shape):
        return [-1] * len(data_shape)


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
    shape_modificator = True

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
            'stft_padded': BoolField(optional=True, default=True,
                                     description="Enables padding at the end of input signal for STFT"),
            'stft_boundary': StringField(choices=['even', 'odd', 'constant', 'zeros'], optional=True, default='zeros',
                                         description="Specifies how to generate boundary values for STFT"),
            'do_transpose': BoolField(optional=True, default=True, description="Enables input transpose"),
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
        self.stft_padded = self.get_value_from_config('stft_padded')
        self.stft_boundary = self.get_value_from_config('stft_boundary')
        self.do_transpose = self.get_value_from_config('do_transpose')

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
        if self.sample_rate is None and image.metadata.get('sample_rate') is None:
            raise RuntimeError(
                'Operation "{}" failed: required "sample rate" in metadata.'.format(self.__provider__)
            )
        sample_rate = self.sample_rate if self.sample_rate is not None else image.metadata['sample_rate']
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
                           noverlap=self.hop_length, nfft=self.n_fft,
                           padded=self.stft_padded, boundary=self.stft_boundary)
        x *= sum(self.window)

        # get power spectrum
        if self.mag_power != 1.0:
            x = np.abs(x) ** self.mag_power

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
        if self.do_transpose:
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
