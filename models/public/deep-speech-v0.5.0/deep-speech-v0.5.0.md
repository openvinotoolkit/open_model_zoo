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
2. Audio Feature, name: `image_node`, shape: [1x16x19x26], format: [BxSxCxI],
   where:

    - B - batch size
    - S - number of steps
    - C - number of context
    - I - number of input

### Converted Model

1. Audio Feature, name: `image_node`, shape: [1x16x19x26], format: [BxSxCxI],
   where:

    - B - batch size
    - S - number of steps
    - C - number of context
    - I - number of input

    * Note: Needs to acquire the mfcc feature for audio.

## Output

### Original Model

The output contain predicted sentence for audio(speech) which decode from ctc beam search decoder. The model was trained on the LibriSpeech clean test corpus for 16-bit, 16 kHz, mono-channel WAVE audio files.


### Converted Model

The output contain predicted sentence for audio(speech) which decode from ctc beam search decoder. The model was trained on the LibriSpeech clean test corpus for 16-bit, 16 kHz, mono-channel WAVE audio files.

## Legal Information

[https://raw.githubusercontent.com/tensorflow/models/master/LICENSE]()
