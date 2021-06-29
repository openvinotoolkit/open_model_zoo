# mobilenet-v2

## Use Case and High-Level Description

[MobileNet V2](https://arxiv.org/abs/1801.04381)

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.876         |
| MParams           | 3.489         |
| Source framework  | Caffe\*       |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 71.218%|
| Top 5  | 90.178%|

## Input

### Original Model

Image, name: `data`,  shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values: [103.94, 116.78, 123.68], scale value: 58.8235294117647.

### Converted Model

Image, name: `data`,  shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in a range [0, 1]

### Converted Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in a range [0, 1]

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
[license](https://raw.githubusercontent.com/shicai/MobileNet-Caffe/26a8b8c0afb6114a07c1c9e4f550e4e0dd8cced1/LICENSE):

```
BSD 3-Clause License

Copyright (c) 2017-, Shicai Yang
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
