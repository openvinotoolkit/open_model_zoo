# ForwardTacotron

## Use Case and High-Level Description

ForwardTacotron is model for text to speech task originally trained on PyTorch\*
then converted to ONNX\* format. Forward Tacotron is the model for mel-spectrogramm regression from text.
For details see [paper](https://arxiv.org/pdf/1703.10135.pdf),[paper](https://arxiv.org/pdf/1905.09263.pdf), [repository](https://github.com/as-ideas/ForwardTacotron).

### Steps to Reproduce PyTorch to ONNX Conversion
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
4. Copy provided scripts `forward_to_onnx.py` to ForwardTacotron root directory.

5. Create pretrained directory and copy the best model to it.
```sh
mkdir pretrained
cp checkouts/ljspeech_tts.forward/forward_<iteration>k.pyt pretrained
```
6. Run provided script for conversion ForwardTacotron to onnx format
```sh
python3 forward_to_onnx.py --force_cpu --tts_weights pretrained/forward_<iteration>K.pyt
```
Notes:
   1. By the reason of unsupported operation in ONNX, the model is divided into two parts: `forward_tacotron_duration_prediction.onnx, forward_tacotron_regression.onnx`.
   2. We stoped training of the Tacotron model in 183K iteration for aligment generation and stoped ForwardTacotron training in 290K iteration.

## ONNX Models
We provide pretrained models in ONNX format for user convenience.

## Specification

| Metric           | Value              |
|------------------|--------------------|
| Type             | Mean Opinion Score |
| GFlops           | -                  |
| MParams          | -                  |
| Source framework | PyTorch\*          |

## Accuracy

Subjective

## Performance


### ForwardTacotron duration predictor

## Input

Sequence, name: `input_seq`, shape: [1x241], format: [BxC]
where:

   - B - batch size
   - C - number of symbols in sequence (letters or phonemes)

## Output

1. Duration for input symbols, name: `duration`, shape: [1, 241, 1], format [BxCxH]. Contains predicted duration for each of the symbol in sequence.
   - B - batch size
   - C - number of symbols in sequence (letters or phonemes)
   - H - empty dimension.
2. Processed embeddings, name: `embeddings`, shape: [1, 241, 512], format [BxCxH]. Contains processed embeddings for each symbol in sequence.
   - B - batch size
   - C - number of symbols in sequence (letters or phonemes)
   - H - height of the intermediate feature map.
### ForwardTacotron regression

## Input

Processed embeddigs aligned by durations, name: `data`, shape: [1x805x512], format: [BxTxC]
where:

   - B - batch size
   - T - time in mel-spectrogramm
   - C - processed embedding dimention

## Output

Mel-spectorgramm, name: `mel`, shape: [80x805], format: [CxT]
where:

   - T - time in mel-spectrogramm
   - C - number of mels in mel-spectrogramm

## Legal Information

The original model is distributed under the following
[license](https://github.com/as-ideas/ForwardTacotron/blob/78789c1aa845057bb2f799e702b1be76bf7defd0/LICENSE)

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
