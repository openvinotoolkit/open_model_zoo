# forward-tacotron (composite)

## Use Case and High-Level Description

ForwardTacotron is a model for the text-to-speech task originally trained in PyTorch\*
then converted to ONNX\* format. The model was trained on LJSpeech dataset. ForwardTacotron performs mel-spectrogram regression from text.
For details see [paper](https://arxiv.org/abs/1703.10135), [paper](https://arxiv.org/abs/1905.09263), [repository](https://github.com/as-ideas/ForwardTacotron).

## ONNX Models
We provide pre-trained models in ONNX format for user convenience.

### Steps to Reproduce training in PyTorch and Conversion to ONNX
Model is provided in ONNX format, which was obtained by the following steps.

1. Clone the original repository
```sh
git clone https://github.com/as-ideas/ForwardTacotron
cd ForwardTacotron
```
2. Checkout the commit that the conversion was tested on:
```sh
git checkout 78789c1aa845057bb2f799e702b1be76bf7defd0
```
3. Follow README.md and train ForwardTacotron model.
4. Copy provided script `forward_to_onnx.py` to ForwardTacotron root directory.
5. Run provided script for conversion ForwardTacotron to onnx format
```sh
python3 forward_to_onnx.py --tts_weights checkpoints/ljspeech_tts.forward/fast_speech_step<iteration>K_weights.pyt
```
Notes:
   1. Since ONNX doesn't support the build_index operation from PyTorch pipeline, the model is divided into two parts: `forward_tacotron_duration_prediction.onnx, forward_tacotron_regression.onnx`.
   2. We stopped training of the Tacotron model in 183K iteration for alignment generation and stopped ForwardTacotron training in 290K iteration.

## Composite model specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Source framework                | PyTorch\*                                 |

### Accuracy

Subjective

## forward-tacotron-duration-prediction model specification

The forward-tacotron-duration-prediction model accepts preprocessed text (see text_to_sequence in [repository](https://github.com/as-ideas/ForwardTacotron/blob/78789c1aa845057bb2f799e702b1be76bf7defd0/utils/text/__init__.py)) and produces processed embeddings and
duration in time for every processed embedding.

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| GOPs                            | 6.66                                      |
| MParams                         | 13.81                                     |

### Input

Sequence, name: `input_seq`, shape: `1, 241`, format: `B, C`, where:

- `B` - batch size
- `C` - number of symbols in sequence (letters or phonemes)

### Output

1. Duration for input symbols, name: `duration`, shape: `1, 241, 1`, format `B, C, H`. Contains predicted duration for each of the symbol in sequence.

   - `B` - batch size
   - `C` - number of symbols in sequence (letters or phonemes)
   - `H` - empty dimension

2. Processed embeddings, name: `embeddings`, shape: `1, 241, 512`, format `B, C, H`. Contains processed embeddings for each symbol in sequence.

   - `B` - batch size
   - `C` - number of symbols in sequence (letters or phonemes)
   - `H` - height of the intermediate feature map

## forward-tacotron-regression model specification

The forward-tacotron-regression model accepts aligned by duration processed embeddings (for example: if duration is [2, 3] and processed embeddings is [[1, 2], [3, 4]], aligned embeddings is [[1, 2], [1, 2], [1,2], [3, 4], [3, 4]]) and produces mel-spectrogram.

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| GOPs                            | 4.91                                      |
| MParams                         | 3.05                                      |

### Input

Processed embeddings aligned by durations, name: `data`, shape: `1, 805, 512`, format: `B, T, C`, where:

- `B` - batch size
- `T` - time in mel-spectrogram
- `C` - processed embedding dimension

### Output

Mel-spectrogram, name: `mel`, shape: `80, 805`, format: `C, T`, where:

- `T` - time in mel-spectrogram
- `C` - number of mels in mel-spectrogram

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/downloader/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>
```

An example of using the Model Converter:
```
python3 <omz_dir>/tools/downloader/converter.py --name <model_name>
```

## Legal Information

The original model is distributed under the following
[license](https://github.com/as-ideas/ForwardTacotron/blob/78789c1aa845057bb2f799e702b1be76bf7defd0/LICENSE):

```
MIT License

Copyright (c) 2020 Axel Springer AI. All rights reserved.
Copyright (c) 2019 fatchord (https://github.com/fatchord)

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
