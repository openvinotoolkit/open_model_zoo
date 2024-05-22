"""
 Copyright (c) 2020-2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

This file is based in fatchord_version.py from https://github.com/fatchord/WaveRNN,
commit 3595219b2f2f5353f0867a7bb59abcb15aba8831 on Nov 27, 2019
"""

import math

import numpy as np


# for WaveRNN approach (https://github.com/fatchord/WaveRNN), first step before upsample
def pad_tensor(x, pad, side='both'):
    # NB - this is just a quick method i need right now
    # i.e., it won't generalise to other shapes/dims
    b, t, c = x.shape
    total = t + 2 * pad if side == 'both' else t + pad
    padded = np.zeros((b, total, c), dtype=float)
    if side in ('before', 'both'):
        padded[:, pad:pad + t, :] = x
    elif side == 'after':
        padded[:, :t, :] = x
    return padded


# https://github.com/fatchord/WaveRNN
def fold_with_overlap(x, target, overlap):
    ''' Fold the tensor with overlap for quick batched inference.
        Overlap will be used for crossfading in xfade_and_unfold()

    Args:
        x (tensor)    : Upsampled conditioning features.
                        shape=(1, timesteps, features)
        target (int)  : Target timesteps for each index of batch
        overlap (int) : Timesteps for both xfade and rnn warmup

    Return:
        (tensor) : shape=(num_folds, target + 2 * overlap, features)

    Details:
        x = [[h1, h2, ... hn]]

        Where each h is a vector of conditioning features

        Eg: target=2, overlap=1 with x.size(1)=10

        folded = [[h1, h2, h3, h4],
                  [h4, h5, h6, h7],
                  [h7, h8, h9, h10]]
    '''

    _, total_len, features = x.shape

    # Calculate variables needed
    num_folds = (total_len - overlap) // (target + overlap)
    if num_folds < 1:
        raise ValueError('Too short mel-spectrogram with width {0}. Try longer sentence.'.format(total_len))
    log_2 = math.log2(num_folds)
    optimal_batch_sz = 2 ** int(math.ceil(log_2))

    offset = 1 if optimal_batch_sz > 1 else 0
    target = (total_len - overlap) // (optimal_batch_sz - offset) - overlap
    num_folds = (total_len - overlap) // (target + overlap)
    if num_folds * (overlap + target) + overlap == total_len:
        overlap += 1
        target = (total_len - overlap) // (optimal_batch_sz - offset) - overlap

    while target < overlap:
        overlap = overlap // 2
        target = (total_len - overlap) // (optimal_batch_sz - offset) - overlap
        if optimal_batch_sz * (overlap + target) + overlap == total_len:
            target = (total_len - overlap) // optimal_batch_sz - overlap

    num_folds = (total_len - overlap) // (target + overlap)

    extended_len = num_folds * (overlap + target) + overlap
    remaining = total_len - extended_len
    # Pad if some time steps poking out
    if remaining != 0:
        num_folds += 1
        padding = target + 2 * overlap - remaining
        x = pad_tensor(x, padding, side='after')

    folded = np.zeros((num_folds, target + 2 * overlap, features), dtype=float)


    # Get the values for the folded tensor
    for i in range(num_folds):
        start = i * (target + overlap)
        end = start + target + 2 * overlap
        folded[i] = x[:, start:end, :]

    return folded, (target, overlap)


# https://github.com/fatchord/WaveRNN
def xfade_and_unfold(y, overlap):
    ''' Applies a crossfade and unfolds into a 1d array.

    Args:
        y (ndarry)    : Batched sequences of audio samples
                        shape=(num_folds, target + 2 * overlap)
                        dtype=float
        overlap (int) : Timesteps for both xfade and rnn warmup

    Return:
        (ndarry) : audio samples in a 1d array
                   shape=(total_len)
                   dtype=float

    Details:
        y = [[seq1],
             [seq2],
             [seq3]]

        Apply a gain envelope at both ends of the sequences

        y = [[seq1_in, seq1_target, seq1_out],
             [seq2_in, seq2_target, seq2_out],
             [seq3_in, seq3_target, seq3_out]]

        Stagger and add up the groups of samples:

        [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

    '''
    num_folds, length = y.shape
    target = length - 2 * overlap
    total_len = num_folds * (target + overlap) + overlap

    # Need some silence for the rnn warmup
    silence_len = overlap // 2
    fade_len = overlap - silence_len
    silence = np.zeros((silence_len), dtype=np.float64)
    linear = np.ones((silence_len), dtype=np.float64)

    # Equal power crossfade
    t = np.linspace(-1, 1, fade_len, dtype=np.float64)
    a = 0.3
    fade_in = np.sqrt(a * (1 + t))
    fade_out = np.sqrt((1.0 - a) * (1 - t))

    # Concat the silence to the fades
    fade_in = np.concatenate([silence, fade_in])
    fade_out = np.concatenate([linear, fade_out])

    # Apply the gain to the overlap samples
    y[:, :overlap] *= fade_in
    y[:, -overlap:] *= fade_out

    unfolded = np.zeros((total_len), dtype=np.float64)

    # Loop to add up all the samples
    for i in range(num_folds):
        start = i * (target + overlap)
        end = start + target + 2 * overlap
        unfolded[start:end] += y[i]

    return unfolded


def get_one_hot(argmaxes, n_classes):
    res = np.eye(n_classes)[np.array(argmaxes).reshape(-1)]
    return res.reshape(list(argmaxes.shape)+[n_classes])


def infer_from_discretized_mix_logistic(params):
    """
    Sample from discretized mixture of logistic distributions
    Args:
        params (Tensor): B x C x T, [C/3,C/3,C/3] = [logit probs, means, log scales]
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    log_scale_min = float(np.log(1e-14))
    assert params.shape[1] % 3 == 0
    nr_mix = params.shape[1] // 3

    # B x T x C
    y = params #np.transpose(params, (1, 0))
    logit_probs = y[:, :nr_mix]

    temp = np.random.uniform(low=1e-5, high=1.0 - 1e-5, size=logit_probs.shape)
    temp = logit_probs - np.log(- np.log(temp))
    argmax = np.argmax(temp, axis=-1)

    one_hot = get_one_hot(argmax, nr_mix).astype(dtype=float)

    means = np.sum(y[:, nr_mix:2 * nr_mix] * one_hot, axis=-1)
    log_scales = np.clip(np.sum(
        y[:, 2 * nr_mix:3 * nr_mix] * one_hot, axis=-1), a_min=log_scale_min, a_max=None)

    u = np.random.uniform(low=1e-5, high=1.0 - 1e-5, size=means.shape)
    x = means + np.exp(log_scales) * (np.log(u) - np.log(1. - u))

    x = np.clip(x, a_min=-1., a_max=1.)

    return x
