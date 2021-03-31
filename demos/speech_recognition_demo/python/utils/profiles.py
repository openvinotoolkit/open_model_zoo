#
# Copyright (C) 2019-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
PROFILES = {
    'mds06x_en': {
        'alphabet': None,  # the default alphabet
        # alpha: Language model weight
        'alpha': 0.75,
        # beta: Word insertion bonus (ignored without LM)
        'beta': 1.85,
        'model_sampling_rate': 16000,
        'frame_window_size_seconds': 32e-3,
        'frame_stride_seconds': 20e-3,
        'mel_num': 40,
        'mel_fmin': 20.,
        'mel_fmax': 4000.,
        'num_mfcc_dct_coefs': 26,
        'num_context_frames': 19,
        'in_state_c': 'previous_state_c',
        'in_state_h': 'previous_state_h',
        'out_state_c': 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/BlockLSTM/TensorIterator.2',
        'out_state_h': 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/BlockLSTM/TensorIterator.1',
        'in_data': 'input_node',
        'out_data': 'logits',
        'log_probs': False,
    },
    'mds07x_en': {
        'alphabet': None,  # the default alphabet
        'alpha': 0.93128901720047,
        'beta': 1.1834137439727783,
        'model_sampling_rate': 16000,
        'frame_window_size_seconds': 32e-3,
        'frame_stride_seconds': 20e-3,
        'mel_num': 40,
        'mel_fmin': 20.,
        'mel_fmax': 8000.,
        'num_mfcc_dct_coefs': 26,
        'num_context_frames': 19,
        'in_state_c': 'previous_state_c',
        'in_state_h': 'previous_state_h',
        'out_state_c': 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/BlockLSTM/TensorIterator.2',
        'out_state_h': 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/BlockLSTM/TensorIterator.1',
        'in_data': 'input_node',
        'out_data': 'logits',
        'log_probs': False,
    },
}
PROFILES['mds08x_en'] = PROFILES['mds07x_en']
PROFILES['mds09x_en'] = PROFILES['mds07x_en']
