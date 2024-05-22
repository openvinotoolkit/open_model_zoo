#
# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
PROFILES = {
    'mds06x_en': {
        # === MFCC feature extraction parameters ===
        # model_sampling_rate (float, in Hz)
        'model_sampling_rate': 16000,
        # frame_window_size_seconds (float, in seconds)
        'frame_window_size_seconds': 32e-3,
        # frame_stride_seconds (float, in seconds)
        'frame_stride_seconds': 20e-3,
        # mel_num (int), number of Mel-spectrum filter banks
        'mel_num': 40,
        # mel_fmin (float, in Hz), Mel-spectrum filter banks range
        'mel_fmin': 20.,
        # mel_fmax (float, in Hz), Mel-spectrum filter banks range
        'mel_fmax': 4000.,
        # num_mfcc_dct_coefs (int)
        'num_mfcc_dct_coefs': 26,  # affects RNN stage as well

        # === RNN parameters ===
        # num_context_frames (int)
        'num_context_frames': 19,
        # in_state_c, in_state_h, out_state_c, out_state_h, in_data, out_data (str), IR node names
        'in_state_c': 'previous_state_c:0',
        'in_state_h': 'previous_state_h:0',
        'out_state_c': 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd:0',
        'out_state_h': 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd_1:0',
        'in_data': 'input_node',
        'out_data': 'logits:0',  # Despite being named logits, output is probabilities after softmax

        # === CTC decoder and LM parameters ===
        # log_probs (bool), True is input data contains base e log(probabilities), False if simply probabilities.
        'log_probs': False,
        # alphabet (str or list(str)), alphabet matching the model:
        #     str = filename of a text file with the alphabet (excluding separator=blank symbol)
        #     list(str) = the alphabet itself (expluding separator=blank symbol)
        'alphabet': list(" abcdefghijklmnopqrstuvwxyz'"),
        # alpha (float), language model weight relative to audio model
        'alpha': 0.75,
        # beta (float), word insertion bonus to counteract LM's tendency to prefer fewer words (ignored without LM)
        'beta': 1.85,
    },
}

PROFILES['mds07x_en'] = PROFILES['mds06x_en'].copy()
PROFILES['mds07x_en'].update({
    'mel_fmax': 8000.,
    'alpha': 0.93128901720047,
    'beta': 1.1834137439727783,
})

PROFILES['mds09x_en'] = PROFILES['mds08x_en'] = PROFILES['mds07x_en']
