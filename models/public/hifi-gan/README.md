# wavernn (composite)

## Use Case and High-Level Description

HiFi-GAN is a model for the text-to-speech task originally trained in PyTorch\*
then converted to ONNX\* format. The model was trained on LJSpeech dataset.
HiFi-GAN performs waveform regression from mel-spectrogram.
For details see [paper](https://arxiv.org/abs/2010.05646), [repository](https://github.com/jik876/hifi-gan).

## ONNX Models

We provide pre-trained models in ONNX format for user convenience.

## Composite model specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Source framework                | PyTorch\*                                 |

### Accuracy

Subjective

## HiFi-GAN model specification

The HiFi-GAN model accepts mel-spectrogram and produces normalized raw audio form.

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| GOPs                            | 90.82                                      |
| MParams                         | 13.93                                       |

### Input

Mel-spectrogram, name: `mel`, shape: `1, 80, 128`, format: `B, C, T`, where:

- `B` - batch size
- `T` - time in mel-spectrogram
- `C` - number of mels in mel-spectrogram

### Output

Processed mel-spectrogram, name: `audio`, shape: `1, 1, 32768`, format: `B, C, T`, where:

   - `B` - batch size
   - `T` - time in audio (equal to `time in mel spectrogram` * `hop_length`)
   - `C` - extra dimension (equal to one).


## Legal Information

The original model is distributed under the following
[license](https://github.com/jik876/hifi-gan/blob/master/LICENSE)

```
MIT License

Copyright (c) 2020 Jungil Kong

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
