# wav2vec2-base

## Use Case and High-Level Description

Wav2Vec2.0-base is a model, which pre-trained to learn speech representations on unlabeled data as described in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) paper and fine-tuned for speech recognition task with a Connectionist Temporal Classification (CTC) loss on LibriSpeech dataset containing 960 hours of audio.
The model is composed of a multi-layer convolutional feature encoder which takes as input raw audio and outputs latent speech representations, then fed to a Transformer to build representations capturing information from the entire sequence. For base model Transformer consists of 12 transformer layers and has 768 as feature dimension.
For details please also check [repository](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20) and [model card](https://huggingface.co/facebook/wav2vec2-base-960h).

## Specification

| Metric           | Value              |
| ---------------- | ------------------ |
| Type             | Speech recognition |
| GFLOPs           | 26.843             |
| MParams          | 94.3965            |
| Source framework | PyTorch\*          |

## Accuracy

| Metric                       | Value |
| ---------------------------- | ----- |
| WER @ Librispeech test-clean | 3.39% |

### Input

#### Original model

Normalized audio signal, name - `inputs`,  shape - `B, N`, format is `B, N`, where:

- `B` - batch size
- `N` - sequence length

Model is dynamic and can working with different shapes of input.

**NOTE**: Model expects 16-bit, 16 kHz, mono-channel WAVE audio as input data.

#### Converted model

The converted model has the same parameters as the original model.

### Output

#### Original model

Per-token probabilities (after LogSoftmax) for every symbol in the alphabet, name - `logits`,  shape - `B, N, 32`, output data format is `B, N, C`, where:

- `B` - batch size
- `N` - number of recognized tokens
- `C` - alphabet size

`B` and `N` dimensions can take different values, because model is dynamic. Alphabet size `C` is static and equals 32.
Model alphabet: "[pad]", "[s]", "[/s]", "[unk]", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R", "D", "L", "U", "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z", where:

- `[pad]` - padding token used as CTC-blank label
- `[s]`- start of string
- `[/s]` - end of string
- `[unk]` - unknown symbol
- `|` - whitespace symbol used as separator between words.

#### Converted model

The converted model has the same parameters as the original model.

## Download a Model and Convert it into OpenVINO™ IR Format

You can download models and if necessary convert them into OpenVINO™ IR format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
omz_downloader --name <model_name>
```

An example of using the Model Converter:
```
omz_converter --name <model_name>
```

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Speech Recognition Wav2Vec Python\* Demo](../../../demos/speech_recognition_wav2vec_demo/python/README.md)

## Legal Information

The original model is distributed under the following [license](https://raw.githubusercontent.com/pytorch/fairseq/master/LICENSE).
```
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
