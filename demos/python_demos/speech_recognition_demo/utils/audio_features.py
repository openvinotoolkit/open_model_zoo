#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import librosa


def samples_to_melspectrum(samples, sampling_rate, window_size, stride, n_mels, fmin, fmax):
    window_size, stride = round(window_size), round(stride)

    # window_size must be a power of 2 to match tf:
    if not(window_size > 0  and  (window_size - 1) & window_size == 0):
        raise ValueError("window_size(ms)*sampling_rate(kHz) must be a power of two")

    spec = np.abs(librosa.core.spectrum.stft(
        samples,
        n_fft=window_size, hop_length=stride, win_length=window_size,
        center=False, window='hann', pad_mode='reflect',
    ))
    # match tf: norm=None
    mel_basis = librosa.filters.mel(
        sampling_rate, window_size,
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
