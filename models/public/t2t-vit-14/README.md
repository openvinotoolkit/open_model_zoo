# t2t-vit-14

## Use Case and High-Level Description

The `t2t-vit-14` model is a variant of the Tokens-To-Token Vision Transformer(T2T-ViT) pre-trained on ImageNet dataset for image classification task. T2T-ViT progressively tokenize the image to tokens and has an efficient backbone. T2T-ViT consists of two main components: 1) a layer-wise "Tokens-to-Token module" to model the local structure information of the image and reduce the length of tokens progressively; 2) an efficient "T2T-ViT backbone" to draw the global attention relation on tokens from the T2T module. The model has 14 transformer layers in T2T-ViT backbone with 384 hidden dimensions.

More details provided in the [paper](https://arxiv.org/abs/2101.11986) and [repository](https://github.com/yitu-opensource/T2T-ViT).

## Specification

| Metric                          | Value          |
|---------------------------------|----------------|
| Type                            | Classification |
| GFlops                          | 9.5451         |
| MParams                         | 21.5498        |
| Source framework                | PyTorch\*      |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 81.44% |
| Top 5  | 95.66% |

## Input

### Original Model

Image, name: `image`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values - [123.675, 116.28, 103.53], scale values - [58.395, 57.12, 57.375].

### Converted Model

Image, name: `image`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name: `probs`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - vector of probabilities for all dataset classes in logits format

### Converted Model

Object classifier according to ImageNet classes, name: `probs`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - vector of probabilities for all dataset classes in logits format

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

* [Classification Python\* Demo](../../../demos/classification_demo/python/README.md)

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/yitu-opensource/T2T-ViT/main/LICENSE):

```
The Clear BSD License

Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Shanghai Yitu Technology Co., Ltd. nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY SHANGHAI YITU TECHNOLOGY CO., LTD. AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SHANGHAI YITU TECHNOLOGY CO., LTD. OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
