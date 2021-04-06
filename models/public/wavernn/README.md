# wavernn (composite)

## Use Case and High-Level Description

WaveRNN is a model for the text-to-speech task originally trained in PyTorch\*
then converted to ONNX\* format. The model was trained on LJSpeech dataset.
WaveRNN performs waveform regression from mel-spectrogram.
For details see [paper](https://arxiv.org/abs/1703.10135), [repository](https://github.com/as-ideas/ForwardTacotron).

## ONNX Models

We provide pre-trained models in ONNX format for user convenience.

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
3. Follow README.md and preprocess LJSpeech dataset.
4. Copy provided script `wavernn_to_onnx.py` to ForwardTacotron root directory, and apply git patch `0001-Added-batch-norm-fusing-to-conv-layers.patch`.
5. Download WaveRNN model from https://github.com/fatchord/WaveRNN/tree/master/pretrained/ and extract in to pre-trained directory.
```sh
mkdir pretrained
wget https://raw.githubusercontent.com/fatchord/WaveRNN/master/pretrained/ljspeech.wavernn.mol.800k.zip
unzip ljspeech.wavernn.mol.800k.zip -d pretrained && mv pretrained/latest_weights.pyt pretrained/wave_800K.pyt
```
6. Run provided script for conversion WaveRNN to onnx format
```sh
python3 wavernn_to_onnx.py --mel <path_to_preprocessed_dataset>/mel/LJ008-0254.npy --voc_weights pretrained/wave_800K.pyt --hp_file hparams.py --batched
```
Note: by the reason of autoregressive nature of the network, the model is divided into two parts: `wavernn_upsampler.onnx, wavernn_rnn.onnx`. The first part expands feature map by the time dimension, and the second one iteratively processes every column in expanded feature map.

## Composite model specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Source framework                | PyTorch\*                                 |

### Accuracy

Subjective

## wavernn-upsampler model specification

The wavernn-upsampler model accepts mel-spectrogram and produces two feature map: the first one expands mel-spectrogram in one step using Upsample layer and sequence of convolutions, and the second one expands mel-spectrogram in three steps using sequence of Upsample layers and of convolutions.

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| GOPs                            | 0.37                                      |
| MParams                         | 0.4                                       |

### Input

Mel-spectrogram, name: `mels`, shape: `1, 200, 80`, format: `B, T, C`, where:

- `B` - batch size
- `T` - time in mel-spectrogram
- `C` - number of mels in mel-spectrogram

### Output

1. Processed mel-spectrogram, name: `aux`, shape: `1, 53888, 128`, format: `B, T, C`, where:

   - `B` - batch size
   - `T` - time in audio (equal to `time in mel spectrogram` * `hop_length`)
   - `C` - number of features in processed mel-spectrogram.

2. Upsampled and processed (by time) mel-spectrogram, name: `upsample_mels`, shape: `1, 55008, 80`, format: `B, T', C`, where:

   - `B` - batch size
   - `T'` - time in audio padded with number of samples for crossfading between batches
   - `C` - number of mels in mel-spectrogram

## wavernn-rnn model specification

The wavernn-rnn model accepts two feature maps from wavernn-upsampler and produces parameters for mixture of logistics distribution that is used for audio regression by B samples per forward step, where B is batch size.

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| GOps                            | 0.06                                      |
| MParams                         | 3.83                                      |

### Input

1. Time slice in `upsampled_mels`, name: `m_t`, shape: `B, 80`
2. Time/space slice in `aux`, name: `a1_t`, shape: `B, 32`, where second dimension is 32 = aux.shape[1] / 4
3. Time/space slice in `aux`, name: `a2_t`, shape: `B, 32`, where second dimension is 32 = aux.shape[1] / 4
4. Time/space slice in `aux`, name: `a3_t`, shape: `B, 32`, where second dimension is 32 = aux.shape[1] / 4
5. Time/space slice in `aux`, name: `a4_t`, shape: `B, 32`, where second dimension is 32 = aux.shape[1] / 4
6. Hidden state for GRU layers in autoregression, name: `h1.1`, shape: `B, 512`
7. Hidden state for GRU layers in autoregression, name: `h2.1`, shape: `B, 512`
8. Previous prediction for autoregression (initially equal to zero), name: `x`, shape: `B, 1`

Note: `B` - batch size.

### Output

1. Hidden state for GRU layers in autoregression, name: `h1`, shape: `B, 512`
2. Hidden state for GRU layers in autoregression, name: `h2`, shape: `B, 512`
3. Parameters for mixture of logistics distribution, name: `logits`, shape: `B, 30`. Can be divided to parameters of mixture of logistic distributions: probabilities = logits[:, :10], means = logits[:, 10:20], scales = logits[:, 20:30]

Note: `B` - batch size.

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
