# ForwardTacotron + WaveRNN

## Use Case and High-Level Description

WaveRNN models for text to speech task  originally trained on PyTorch\*
then converted to ONNX\* format. Forward Tacotron is the model for mel-spectrogramm regression from text.
For details see [paper](https://arxiv.org/pdf/1703.10135.pdf),[paper](https://arxiv.org/pdf/1905.09263.pdf), [repository](https://github.com/as-ideas/ForwardTacotron).
WaveRNN is the model for waveform regression from mel-spectrogramm.
For details see [paper](https://arxiv.org/pdf/1703.10135.pdf), [repository](https://github.com/as-ideas/ForwardTacotron).

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
4. Copy provided script `wavernn_to_onnx.py` to ForwardTacotron root directory, and script `fatchord_version_fused.py` to ForwardTacotron/models directory.
5. Download WaveRNN model from https://github.com/fatchord/WaveRNN/tree/master/pretrained/ and extract in to pretrained directory.
```sh
mkdir pretrained
wget https://raw.githubusercontent.com/fatchord/WaveRNN/master/pretrained/ljspeech.wavernn.mol.800k.zip
unzip ljspeech.wavernn.mol.800k.zip -d pretrained && mv pretrained/latest_weights.pyt pretrained/wave_800K.pyt
```
6. Run provided script for conversion WaveRNN to onnx format
```sh
python3 wavernn_to_onnx.py --mel <path_to_some_mel_spectrogramm_file_from_train_directory>.npy --voc_weights pretrained/wave_800K.pyt --hp_file hparams.py --force_cpu --batched
```
Note: by the reason of autoregressive nature of the network, the model is divided into two parts: `wavernn_upsampler.onnx, wavernn_rnn.onnx`.

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

### WaveRNN upsampler

## Input

Mel-spectrogramm, name: `mels`, shape: [1x793x80], format: [BxTxC]
where:

   - B - batch size
   - T - time in mel-spectrogramm
   - C - number of mels in mel-spectrogramm

## Output

1. Processed mel-spectrogram, name: `aux`, shape: [1x102025x128], format: [BxTxC]
where:
   - B - batch size
   - T - time in audio (equal to `time in mel spectrogramm` * `hop_lenght`)
   - C - number of features in processed mel-spectrogramm.

2. Upsampled and processed (by time) mel-spectrogramm, name: `upsample_mels`, shape: [1x103125x80], format: [BxT'xC]
where:
   - B - batch size
   - T' - time in audio padded with number of samples for crossfading between batches
   - C - number of mels in mel-spectrogramm


### WaveRNN autoregression part

## Input
1. Time slice in `upsampled_mels`, name: `m_t`. Shape: [Bx80]
2. Time/space slices in `aux`, name: `a1_t`, `a2_t`, `a3_t`,`a4_t`. Shape: [Bx32]. Second dimention is 32 = aux.shape[1] / 4
3. Hidden states for GRU layers in autoregression, name `h1.1`, `h2.1`. Shape: [Bx512].
4. Previous prediction for autoregression (initially equal to zero), name: `x`. Shape: [Bx1]

Note: B - batch size.

## Output
1. Hidden states for GRU layers in autoregression, name `h1`, `h2`. Shape: [Bx512].
2. Parameters for mixture of logistics distribution, name: `logits`. Shape: [Bx30]. Can be divided to parameters of mixture of logistic distributions: probabilities = logits[:, :10], means = logits[:, 10:20], scales = logits[:, 20:30].

Note: B - batch size.

## Legal Information

The original model is distributed under the following
[license](https://github.com/fatchord/WaveRNN/blob/master/LICENSE.txt)

```
MIT License

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
