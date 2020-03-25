# colorization-v2-norebal

## Use Case and High-Level Description

The `colorization-v2-norebal` model is one of the [colorization](https://arxiv.org/abs/1603.08511)
group of models designed to perform image colorization. For details
about this family of models, check out the [repository](https://github.com/richzhang/colorization).

This model differs from model `colorization-v2` in that metrics did not take into account
balancing of rare classes during training.

Model consumes as input L-channel of LAB-image.
Model give as output predict A- and B-channels of LAB-image.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Colorization  |
| GFLOPs            | -             |
| MParams           | -             |
| Source framework  | Caffe\*       |

## Accuracy

The accuracy metrics calculated on ImageNet
validation dataset using [VGG16](https://arxiv.org/abs/1409.1556) caffe
model and colorization as preprocessing.

For preprocessing `rgb -> gray -> coloriaztion` recieved values:

| Metric         | Value with preprocessing   | Value without preprocessing |
|----------------|-----------------------------|-----------------------------|
| Accuracy top-1 |                      57.24% |                      70.96% |
| Accuracy top-5 |                      80.96% |                      89.88% |

## Performance

## Input

### Original model

Image, name - `data_l`,  shape - `1,1,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is L-channel.
Mean values - 50.

### Converted model

Image, name - `data_l`,  shape - `1,1,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is L-channel.

## Output


### Original model

Image, name - `class8_ab`\*,  shape - `1,2,56,56`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

### Converted model

Image, name - `class8_313_rh`\*,  shape - `1,313,56,56`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

> **NOTE**: `class8_313_rh` layer is in front of `class8_ab` layer,
in order for network to work,
you need to reproduce `class8_ab` layer with the coefficients that
downloaded separately with the model.

## Legal Information
The original model is distributed under the following
[license](https://raw.githubusercontent.com/richzhang/colorization/master/LICENSE):

```
Copyright (c) 2016, Richard Zhang, Phillip Isola, Alexei A. Efros
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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
