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

from ..config import BoolField, NumberField, BaseField, ConfigError, StringField
from ..preprocessor import Preprocessor
from ..utils import UnsupportedPackage

try:
    from scipy.signal import lfilter
except ImportError as error:
    lfilter = UnsupportedPackage('scipy', error.msg)


class SpliceFrame(Preprocessor):
    __provider__ = 'audio_splice'
    shape_modificator = True

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'frames': NumberField(min_value=2, value_type=int, description="Number of frames to splice"),
            'axis': NumberField(optional=True, default=2, description="Axis to splice frames along", value_type=int),
        })
        return parameters

    def configure(self):
        self.frames = self.get_value_from_config('frames')
        self.axis = self.get_value_from_config('axis')

    def process(self, image, annotation_meta=None):
        image.data = self.process_feats(image.data)

        return image

    def process_feats(self, feats):
        feats_init_shape = feats.shape
        seq = [feats]
        for n in range(1, self.frames):
            tmp = np.zeros(feats_init_shape)
            tmp[:, :-n] = feats[:, n:]
            seq.append(tmp)
        return np.concatenate(seq, axis=self.axis)

    def calculate_out_single_shape(self, data_shape):
        feat = np.zeros(data_shape)
        return self.process_feats(feat).shape

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]


class DitherFrame(Preprocessor):
    __provider__ = 'audio_dither'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'use_deterministic_dithering': BoolField(optional=True, default=True,
                                                     description="Deterministic dithering flag"),
            'dither': NumberField(optional=True, default=1e-5, description="Dithering factor", value_type=float),
        })
        return parameters

    def configure(self):
        self.use_deterministic_dithering = self.get_value_from_config('use_deterministic_dithering')
        self.dither = self.get_value_from_config('dither')

    def process(self, image, annotation_meta=None):
        if self.dither > 0 and not self.use_deterministic_dithering:
            image.data += self.dither * np.random.rand(*image.data.shape)
        return image


class PreemphFrame(Preprocessor):
    __provider__ = 'audio_preemph'
    shape_modificator = True

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'preemph': NumberField(optional=True, default=0.97, description="Preemph factor", value_type=float),
        })
        return parameters

    def configure(self):
        self.preemph = self.get_value_from_config('preemph')

    def process(self, image, annotation_meta=None):
        if self.preemph == 0:
            return image
        image.data = np.concatenate(
            (np.expand_dims(image.data[:, 0], axis=0), image.data[:, 1:] - self.preemph *  image.data[:, :-1]), axis=1
        )
        return image

    def calculate_out_single_shape(self, data_shape):
        if self.preemph == 0:
            return data_shape
        data = np.zeros(data_shape)
        np.concatenate((np.expand_dims(data[:, 0], axis=0), data[:, 1:] - self.preemph * data[:, :-1]), axis=1)
        return data.shape

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]


class DitherSpectrum(Preprocessor):
    __provider__ = 'audio_spec_dither'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'use_deterministic_dithering': BoolField(optional=True, default=True,
                                                     description="Deterministic dithering flag"),
            'dither': NumberField(optional=True, default=1e-5, description="Dithering factor", value_type=float),
        })
        return parameters

    def configure(self):
        self.use_deterministic_dithering = self.get_value_from_config('use_deterministic_dithering')
        self.dither = self.get_value_from_config('dither')

    def process(self, image, annotation_meta=None):
        if self.dither > 0 and not self.use_deterministic_dithering:
            image.data = image.data + self.dither ** 2
        return image


class SignalPatching(Preprocessor):
    __provider__ = 'audio_patches'
    shape_modificator = True
    _dynamic_output = False

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'size': NumberField(value_type=int, min_value=1)
        })
        return params

    def configure(self):
        self.size = self.get_value_from_config('size')

    def process(self, image, annotation_meta=None):
        data = np.squeeze(image.data)
        patch_num, rest_size = divmod(np.squeeze(data).shape[0], self.size)
        if rest_size > 0:
            data = np.pad(data, (self.size - rest_size, 0), mode='constant')
            patch_num += 1
            image.metadata['padding'] = self.size - rest_size
        processed_data = np.split(data, patch_num)
        image.data = processed_data
        image.metadata['multi_infer'] = True
        return image

    @property
    def dynamic_result_shape(self):
        return self._dynamic_output

    def calculate_out_shape(self, data_shape):
        return [[self.size]] * len(data_shape)


class ContextWindow(Preprocessor):
    __provider__ = 'context_window'
    shape_modificator = True

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'cw_l': NumberField(value_type=int, min_value=0, description='Context window left'),
            'cw_r': NumberField(value_type=int, min_value=0, description='Context window right'),
            'to_multi_infer': BoolField(optional=True, default=False)
        })
        return params

    def configure(self):
        self.cw_l = self.get_value_from_config('cw_l')
        self.cw_r = self.get_value_from_config('cw_r')
        self.to_multi_infer = self.get_value_from_config('to_multi_infer')

    def process(self, image, annotation_meta=None):
        def process_single(signal):
            borders = (self.cw_l, self.cw_r) if signal.ndim == 1 else ((self.cw_l, self.cw_r), (0, 0))
            return np.pad(signal, borders, mode='edge')
        image.data = (
            process_single(image.data) if not isinstance(image.data, list)
            else [process_single(elem) for elem in image.data]
        )
        image.metadata['context_left'] = self.cw_l
        image.metadata['context_right'] = self.cw_r
        if self.to_multi_infer:
            image.data = list(image.data)
            image.metadata['multi_infer'] = True

        return image

    def calculate_out_single_shape(self, data_shape):
        if self.to_multi_infer:
            return data_shape[1:]
        first_dim = data_shape[0] + self.cw_r + self.cw_l
        new_shape = list(data_shape)
        new_shape[0] = first_dim
        return tuple(new_shape)

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]


class ResampleAudio(Preprocessor):
    __provider__ = 'resample_audio'
    shape_modificator = True

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
        resampled_data = np.zeros(shape=(data.shape[0], int(duration * self.sample_rate)), dtype=float)
        x_old = np.linspace(0, duration, data.shape[1])
        x_new = np.linspace(0, duration, resampled_data.shape[1])
        resampled_data[0] = np.interp(x_new, x_old, data[0])

        image.data = resampled_data
        image.metadata['sample_rate'] = self.sample_rate

        return image

    @staticmethod
    def calculate_out_single_shape(data_shape):
        return [-1] * len(data_shape)

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]


class ClipAudio(Preprocessor):
    __provider__ = 'clip_audio'
    shape_modificator = True
    _dynamic_shape = False

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
                except ValueError as value_err:
                    raise ConfigError("Preprocessor {}: invalid value for 'overlap' - {}."
                                      .format(self.__provider__, overlap)) from value_err
            elif overlap.endswith('samples'):
                try:
                    self.overlap_in_samples = int(overlap[:-7])
                except ValueError as value_err:
                    raise ConfigError("Preprocessor {}: invalid value for 'overlap' - {}."
                                      .format(self.__provider__, overlap)) from value_err
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
            except ValueError as value_err:
                raise ConfigError("Preprocessor {}: invalid value for 'overlap' - {}."
                                  .format(self.__provider__, overlap)) from value_err
            if self.overlap <= 0 or self.overlap >= 1:
                raise ConfigError("Preprocessor {}: invalid value for 'overlap' - {}."
                                  .format(self.__provider__, overlap))

    def _parse_duration(self, duration):
        self.is_duration_in_samples = False
        if isinstance(duration, str):
            if duration.endswith('samples'):
                try:
                    self.duration = int(duration[:-7])
                except ValueError as value_err:
                    raise ConfigError("Preprocessor {}: invalid value for duration - {}."
                                      .format(self.__provider__, duration)) from value_err
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
            except ValueError as value_err:
                raise ConfigError("Preprocessor {}: invalid value for duration - {}."
                                  .format(self.__provider__, duration)) from value_err
            if self.duration <= 0:
                raise ConfigError("Preprocessor {}: duration should be positive value - {}."
                                  .format(self.__provider__, self.duration))

    @property
    def dynamic_result_shape(self):
        return self._dynamic_shape


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
            'per_frame': BoolField(
                optional=True, default=False, description='apply normalization to each frame separately')
        })
        return parameters

    def configure(self):
        self.int16mode = self.get_value_from_config('int16mode')
        self.per_frame = self.get_value_from_config('per_frame')

    def process(self, image, annotation_meta=None):
        sound = image.data
        if self.int16mode:
            sound = sound / np.float32(0x8000)
        else:
            if not self.per_frame:
                sound = (sound - np.mean(sound)) / (np.std(sound) + 1e-15)
            else:
                sound = np.array([(frame - np.mean(frame)) / (np.std(frame) + 1e-15) for frame in sound[0]])

        image.data = sound

        return image


class AddBatch(Preprocessor):
    __provider__ = 'add_batch'
    shape_modificator = True
    _dynamic_shape = False

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
        image.data = self.process_feat(data)
        return image

    def process_feat(self, data):
        data = np.expand_dims(data, 0)
        if self.axis != 0:
            if self.axis > len(data.shape):
                raise RuntimeError(
                    'Operation "{}" failed: Invalid axis {} for shape {}.'.format(self.__provider__, self.axis,
                                                                                  data.shape)
                )
            order = list(range(1, self.axis + 1)) + [0] + list(range(self.axis + 1, len(data.shape)))
            data = np.transpose(data, order)
        return data

    @property
    def dynamic_result_shape(self):
        return self._dynamic_shape

    def calculate_out_single_shape(self, data_shape):
        return self.process_feat(np.zeros(data_shape)).shape

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) for ds in data_shape]


class RemoveDCandDither(Preprocessor):
    __provider__ = 'remove_dc_and_dither'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'alpha': NumberField(value_type=float, min_value=0, max_value=1, description='alpha'),
        })
        return params

    def configure(self):
        self.alpha = self.get_value_from_config('alpha')

        if isinstance(lfilter, UnsupportedPackage):
            lfilter.raise_error(self.__provider__)

    def process(self, image, annotation_meta=None):
        image.data = self.process_feat(image.data)
        return image

    def process_feat(self, input_signal):
        input_signal = lfilter([1, -1], [1, -1 * self.alpha], input_signal)
        dither = np.random.random_sample(len(input_signal)) + np.random.random_sample(len(input_signal)) - 1
        spow = np.std(dither)
        out_signal = input_signal + 1e-6 * spow * dither
        return out_signal


windows = {
    'none': lambda x: np.ones((x,)),
    'hamming': np.hamming,
    'hanning': np.hanning,
    'blackman': np.blackman,
    'bartlett': np.bartlett,
}


class FrameSignalOverlappingWindow(Preprocessor):
    __provider__ = 'frame_signal_overlap'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'frame_len': NumberField(value_type=float, min_value=0, description='frame length'),
            'frame_step': NumberField(value_type=float, min_value=0, description='frame step'),
            'window': StringField(
                optional=True, default='none', description='rolling window function', choices=windows)
        })
        return params

    def configure(self):
        self.frame_len = self.get_value_from_config('frame_len')
        self.frame_step = self.get_value_from_config('frame_step')
        self.window = windows[self.get_value_from_config('window')]

    def process(self, image, annotation_meta=None):
        def rolling_window(signal, window, step=1):
            shape = signal.shape[:-1] + (signal.shape[-1] - window + 1, window)
            strides = signal.strides + (signal.strides[-1],)
            return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)[::step]

        sample_rate = image.metadata.get('sample_rate')
        if sample_rate is None:
            raise RuntimeError('Operation "{}" can\'t resample audio: required original samplerate in metadata.'.
                               format(self.__provider__))
        frame_len = int(round(self.frame_len * sample_rate))
        frame_step = int(round(self.frame_step * sample_rate))
        signal_len = image.data.shape[1]
        if signal_len <= frame_len:
            num_frames = 1
        else:
            num_frames = 1 + int(np.ceil((1.0 * signal_len - frame_len) / frame_step))
        padding = np.zeros((1, int((num_frames - 1) * frame_step + frame_len) - signal_len))
        padded_signal = np.concatenate((image.data, padding), axis=1)
        win = self.window(frame_len)
        frames = rolling_window(padded_signal, window=frame_len, step=frame_step)
        image.data = frames * win
        return image


class TruncateBucket(Preprocessor):
    __provider__ = 'truncate_bucket'
    shape_modificator = True
    _dynamic_shapes = True

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({'bucket': NumberField(value_type=int, min_value=1)})
        return params

    def configure(self):
        self.bucket_size = self.get_value_from_config('bucket')

    def process(self, image, annotation_meta=None):
        if image.data.shape[1] < self.bucket_size:
            return image
        rsize = self.bucket_size
        rstart = int((image.data.shape[1] - rsize) / 2)
        image.data = image.data[:, rstart:rstart + rsize]
        return image

    @property
    def dynamic_result_shape(self):
        return self._dynamic_shapes

    def calculate_out_shape_single(self, data_shape):
        if data_shape[1] > self.bucket_size:
            return data_shape[0], self.bucket_size
        return data_shape

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_shape_single(ds) for ds in data_shape]
