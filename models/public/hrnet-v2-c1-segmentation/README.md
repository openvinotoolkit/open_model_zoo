# hrnet-v2-c1-segmentation

## Use Case and High-Level Description

This model is a pair of encoder and decoder. The encoder is HRNetV2-W48 and the decoder is C1 (one convolution module and interpolation).
HRNetV2-W48 is semantic-segmentation model based on architecture described in paper
[High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/abs/1904.04514).
This is PyTorch\* implementation based on retaining high resolution representations throughout the model
and pre-trained on ADE20k dataset.
For details about implementation of model, check out the [Semantic Segmentation on MIT ADE20K dataset in PyTorch](https://github.com/CSAILVision/semantic-segmentation-pytorch) repository.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Segmentation  |
| GFLOPs            | 81.9930       |
| MParams           | 66.4768       |
| Source framework  | PyTorch\*     |

## Accuracy

| Metric         | Original model | Converted model |
| -------------- | -------------- | --------------- |
| Pixel accuracy | 77.69%         | 77.69%          |
| mean IoU       | 33.02%         | 33.02%          |

## Input

### Original Model

Image, name - `image`,  shape - `1, 3, 320, 320`, format is `B, C, H, W`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`. Mean values - [123.675, 116.28, 103.53], scale values - [58.395, 57.12, 57.375].

### Converted Model

Image, name - `input.1`,  shape - `1, 3, 320, 320`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Semantic-segmentation mask according to ADE20k classes, name - `softmax`,  shape - `1, 150, 320, 320`, output data format is `B, C, H, W`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range
- `H` - height
- `W` - width

### Converted Model

Semantic-segmentation mask according to ADE20k classes, name - `softmax`,  shape - `1, 150, 320, 320`, output data format is `B, C, H, W`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range
- `H` - height
- `W` - width

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
[license](https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/LICENSE):

```
BSD 3-Clause License

Copyright (c) 2019, MIT CSAIL Computer Vision
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
