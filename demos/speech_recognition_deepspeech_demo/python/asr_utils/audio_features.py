#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from copy import deepcopy

import numpy as np

# Workaround to import librosa on Linux without installed libsndfile.so
try:
    import librosa
except OSError:
    import sys
    import types
    sys.modules['soundfile'] = types.ModuleType('fake_soundfile')
    import librosa

from asr_utils.pipelines import BlockedSeqPipelineStage


class AudioFeaturesSeqPipelineStage(BlockedSeqPipelineStage):
    def __init__(self, profile):
        self.p = deepcopy(profile)
        sampling_rate = self.p['model_sampling_rate']
        window_len = round(sampling_rate * self.p['frame_window_size_seconds'])
        stride_len = round(sampling_rate * self.p['frame_stride_seconds'])

        # window_size must be a power of 2 to match tf:
        if not (window_len > 0 and (window_len - 1) & window_len == 0):
            raise ValueError("window_size(ms)*sampling_rate(kHz) must be a power of two")

        super().__init__(
            block_len=stride_len, context_len=window_len - stride_len,
            left_padding_len=0, right_padding_len=0,
            padding_shape=(), cut_alignment=False)

    def process_data(self, data, finish=False):
        if data is not None:
            if len(data.shape) == 2 and data.shape[1] != 1 or len(data.shape) not in [1, 2]:
                raise ValueError("Input audio file should be {} kHz mono".format(self.p['model_sampling_rate']/1e3))
            if len(data.shape) == 2:
                data = data.squeeze(axis=1)
        return super().process_data(data, finish=finish)

    def _process_blocks(self, audio, finish=False):
        """
        audio (numpy.ndarray), this buffer is guaranteed to contain data for 1 or more blocks
            (audio.shape[0]>=self._block_len+self._context_len)
        """
        # Cut the buffer to the end of the last frame
        audio_len = audio.shape[0]
        processable_len = audio_len - (audio_len - self._context_len) % self._block_len
        buffer_skip_len = processable_len - self._context_len
        audio = audio[:processable_len]

        # Convert audio data type to float32 if needed
        if np.issubdtype(audio.dtype, np.uint8):
            audio = audio/np.float32(128) - 1  # normalize to -1 to 1, uint8 to float32
        elif np.issubdtype(audio.dtype, np.integer):
            audio = audio/np.float32(32768)  # normalize to -1 to 1, int16 to float32

        melspectrum = samples_to_melspectrum(
            audio,                               # samples
            self.p['model_sampling_rate'],       # sampling_rate
            self._context_len + self._block_len, # window_size
            self._block_len,                     # stride
            n_mels = self.p['mel_num'],
            fmin = self.p['mel_fmin'],
            fmax = self.p['mel_fmax'],
        )
        if self.p['num_mfcc_dct_coefs'] is not None:
            mfcc_features = melspectrum_to_mfcc(melspectrum, self.p['num_mfcc_dct_coefs'])
            return [mfcc_features], buffer_skip_len
        else:
            return [melspectrum], buffer_skip_len


def samples_to_melspectrum(samples, sampling_rate, window_size, stride, n_mels, fmin, fmax):
    window_size, stride = round(window_size), round(stride)

    # window_size must be a power of 2 to match tf:
    if not (window_size > 0 and (window_size - 1) & window_size == 0):
        raise ValueError("window_size(ms)*sampling_rate(kHz) must be a power of two")

    spec = np.abs(librosa.stft(
        samples,
        n_fft=window_size, hop_length=stride, win_length=window_size,
        center=False, window='hann', pad_mode='reflect',
    ))
    # match tf: norm=None
    mel_basis = librosa.filters.mel(
        sr=sampling_rate, n_fft=window_size,
        n_mels=n_mels, fmin=fmin, fmax=fmax,
        norm=None, htk=True,
    )
    # match tf: zero spectrum below fmin/sr*2*(n_fft-1)
    freq_bin_fmin = round(fmin/sampling_rate*2*(window_size-1))
    spec[:freq_bin_fmin+1, :] = 0.

    melspectrum = np.dot(mel_basis, spec)
    return melspectrum


def melspectrum_to_mfcc(melspectrum, dct_coefficient_count):
    # match tf: use np.log() instead of power_to_db() to get correct normalization
    mfcc = librosa.feature.mfcc(
        S=np.log(melspectrum + 1e-30),
        norm='ortho',
        n_mfcc=dct_coefficient_count,
    )
    # match tf: un-correct 0-th bin normalization
    mfcc[0] *= 2**0.5
    return mfcc.T
