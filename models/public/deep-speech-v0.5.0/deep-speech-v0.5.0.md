# deep-speech-v0.5.0

## Use Case and High-Level Description

DeepSpeech is an open source Speech-To-Text engine, using a model trained by machine learning techniques based on [Baidu's Deep Speech research paper](https://arxiv.org/abs/1412.5567). Project DeepSpeech uses Google's [TensorFlow](https://www.tensorflow.org/) to make the implementation easier.

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Speech recognition                        |
| GFlops                          | 416.97                                    |
| MParams                         | 47.225                                    |
| Source framework                | TensorFlow\*                              |

## Performance

## Input

### Original Model

1. Audio, name: `input_samples`, a WAVE file.
2. Audio Feature, name: `input_node`, shape: [1x16x19x26], format: [BxSxCxI],
   where:

    - B - batch size
    - S - number of steps
    - C - number of context
    - I - number of input
    
3. Hidden state, name: `previous_state_h/read`, shape: [1x2048], format: [BxF],
   where: 
    
    - B - batch size
    - F - number of features in the hidden state `h`
    * Note: The initial value of hidden state `h` is zero.

4. Hidden state, name: `previous_state_c/read`, shape: [1x2048], format: [BxF], 
   where: 
   
    - B - batch size
    - F - number of features in the hidden state `c`
    * Note: The initial value of hidden state `c` is zero.

### Converted Model

1. Audio Feature, name: `input_node`, shape: [1x16x19x26], format: [BxSxCxI],
   where:

    - B - batch size
    - S - number of steps
    - C - number of context
    - I - number of input

    > Note: Needs to acquire the mfcc feature for audio.
    
2. Hidden state, name: `previous_state_h/read/placeholder_port_0`, shape: [1x2048], format: [BxF],
   where: 
    
    - B - batch size
    - F - number of features in the hidden state `h`

3. Hidden state, name: `previous_state_c/read/placeholder_port_0`, shape: [1x2048], format: [BxF], 
   where: 
   
    - B - batch size
    - F - number of features in the hidden state `c`

## Output

### Original Model

Output text, name: `Softmax`, Contains predicted sentence for audio(speech) which decode from ctc beam search decoder. The model was trained on the LibriSpeech clean test corpus for 16-bit, 16 kHz, mono-channel WAVE audio files.

Hidden state `h`, name: `lstm_fused_cell/BlockLSTM:6`, Contains hidden state feature from lstm for next iteration.

Hidden state `c`, name: `lstm_fused_cell/BlockLSTM:1`, Contains hidden state feature from lstm for next iteration.

### Converted Model

Output text, name: `Softmax`, Contains predicted sentence for audio(speech) which decode from ctc beam search decoder. The model was trained on the LibriSpeech clean test corpus for 16-bit, 16 kHz, mono-channel WAVE audio files.

Hidden state `h`, name: `lstm_fused_cell/BlockLSTM/TensorIterator.1`, Contains hidden state feature from lstm for next iteration.

Hidden state `c`, name: `lstm_fused_cell/BlockLSTM/TensorIterator.2`, Contains hidden state feature from lstm for next iteration.

## Legal Information

[https://raw.githubusercontent.com/tensorflow/models/master/LICENSE]()
