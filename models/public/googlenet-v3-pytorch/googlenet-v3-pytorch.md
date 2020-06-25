# googlenet-v3-pytorch

## Use Case and High-Level Description

Inception v3 is image classification model pretrained on ImageNet dataset. This
PyTorch implementation of architecture described in the paper ["Rethinking
the Inception Architecture for Computer Vision"](https://arxiv.org/abs/1512.00567) in
TorchVision package (see [here](https://github.com/pytorch/vision)).

The model input is a blob that consists of a single image of "1x3x299x299"
in RGB order.

The model output is typical object classifier for the 1000 different classifications
matching with those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 11.469        |
| MParams           | 23.817        |
| Source framework  | PyTorch\*     |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 77.696%|
| Top 5  | 93.696%|

## Performance

## Input

### Original model

Image, name - `data`, shape - [1x3x299x299], format [BxCxHxW],
   where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `RGB`.

Mean values - [127.5, 127.5, 127.5], scale factor for each channel - 127.5

### Converted model

Image, name - `data`, shape - [1x3x299x299], format [BxCxHxW],
   where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `BGR`.

## Output

Object classifier according to ImageNet classes, name - `prob`, shape - [1,1000] in [BxC] format, where:

- `B` - batch size
- `C` - vector of probabilities for each class in [0, 1] range

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/pytorch/vision/master/LICENSE):

```
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
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
