# mozilla-deepspeech-0.6.1

## Use Case and High-Level Description

"mozilla-deepspeech-0.6.1" is a speech recognition neural network pre-trained by Mozilla
based on DeepSpeech architecture (CTC decoder with beam search and n-gram language model)
with changed neural network topology.

For details on the original DeepSpeech, see paper <https://arxiv.org/abs/1412.5567>.

For details on this model, see <https://github.com/mozilla/DeepSpeech/releases/tag/v0.6.1>.

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Speech recognition                        |
| GFlops per audio frame          | 0.0472                                    |
| GFlops per second of audio      | 2.36                                      |
| MParams                         | 47.2                                      |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric                       | Value      | Parameters                                     |
| ---------------------------- | ---------- | ---------------------------------------------- |
| WER @ Librispeech test-clean | 8.93%      | with LM, beam_width = 32, Python CTC decoder   |
| WER @ Librispeech test-clean | **7.55%**  | with LM, **beam_width = 500**, C++ CTC decoder |

*NB*: beam_width=32 is a low value for a CTC decoder, and was used to achieve reasonable evaluation time with Python CTC decoder in Accuracy Checker.
Increasing beam_width improves WER metric and slows down decoding.  Speech recognition demo has a faster C++ CTC decoder module.

## Performance

## Input

### Original Model

 * **Audio MFCC coefficients**, name: `input_node` , shape: [1x16x19x26], format: [BxNxTxC], where:

    - B - batch size, fixed to 1
    - N - `input_lengths`, number of audio frames in this section of audio
    - T - context frames: along with the current frame, the network expects 9 preceeding frames and 9 succeeding frames. The absent context frames are filled with zeros.
    - C - 26 MFCC coefficients per each frame

   See [`accuracy-check.yml`](accuracy-check.yml) for all audio preprocessing and feature extraction parameters.

 * Number of audio frames, INT32 value, name: `input_lengths`, shape [1].

 * LSTM in-state (*c*) and input (*h*, a.k.a hidden state) vectors. Names: `previous_state_c` and `previous_state_h`, shapes: [1x2048], format: [BxC].

    When splitting a long audio into chunks, these inputs must be fed with the corresponding outputs from the previous chunk.
    Chunk processing order must be from early to late audio positions.


### Converted Model

 * **Audio MFCC coefficients**. Image, name: `input_node` , shape: [1x16x19x26], format: [BxNxTxC], where:

    - B - batch size, fixed to 1
    - N - number of audio frames in this section of audio, fixed to 16
    - T - context frames: along with the current frame, the network expects 9 preceeding frames and 9 succeeding frames. The absent context frames are filled with zeros.
    - C - 26 MFCC coefficients in each frame

    See [`accuracy-check.yml`](accuracy-check.yml) for all audio preprocessing and feature extraction parameters.

 * LSTM in-state and input vectors. Names: `previous_state_c` and `previous_state_h`, shapes: [1x2048], format: [BxC].
When splitting a long audio into chunks, these inputs must be fed with the corresponding outputs from the previous chunk.
Chunk processing order must be from early to late audio positions.

## Output

### Original Model

 * Per-frame probabilities (after softmax) for every symbol in the alphabet, name: `logits`, shape: [16x1x29], format: [NxBxC]

    - N - number of audio frames in this section of audio
    - B - batch size, fixed to 1
    - C - alphabet size, including the CTC blank symbol

    The per-frame probabilities are to be decoded with a CTC decoder.
    The alphabet is: 0 = space, 1...26 = "a" to "z", 27 = apostrophe, 28 = CTC blank symbol.

    *NB*: `logits` is probabilities after softmax, despite its name.

 * LSTM out-state and output vectors. Names: `new_state_c` and `new_state_h`, shapes: [1x2048], format: [BxC]. See Inputs.

### Converted Model

 * Per-frame probabilities (after softmax) for every symbol in the alphabet, name: `logits`, shape: [16x1x29], format: [NxBxC]

    - N - number of audio frames in this section of audio, fixed to 16
    - B - batch size, fixed to 1
    - C - alphabet size, including the CTC blank symbol

    The per-frame probabilities are to be decoded with a CTC decoder.
    The alphabet is: 0 = space, 1...26 = "a" to "z", 27 = apostrophe, 28 = CTC blank symbol.

    *NB*: `logits` is probabilities after softmax, despite its name.

 * LSTM out-state and output vectors. Names:

    - `cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/BlockLSTM/TensorIterator.2` for `new_state_c`
    - `cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/BlockLSTM/TensorIterator.1` for `new_state_h`

   Shapes: [1x2048], format: [BxC].  See the corresponding Inputs.

## Legal Information

The original model is distributed under the
[Mozilla Public License, Version 2.0](https://raw.githubusercontent.com/mozilla/DeepSpeech/master/LICENSE).
A copy of the license is provided in [MPL-2.0-Mozilla-Deepspeech.txt](../licenses/MPL-2.0-Mozilla-Deepspeech.txt).
