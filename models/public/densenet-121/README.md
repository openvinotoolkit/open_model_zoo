# densenet-121

## Use Case and High-Level Description

The `densenet-121` model is one of the [DenseNet](https://arxiv.org/abs/1608.06993)
group of models designed to perform image classification. The authors originally trained the models
on Torch\*, but then converted them into Caffe\* format. All DenseNet models have
been pre-trained on the ImageNet image database. For details about this family of
models, check out the [repository](https://github.com/shicai/DenseNet-Caffe).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 5.724         |
| MParams           | 7.971         |
| Source framework  | Caffe\*       |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 74.42% |
| Top 5  | 92.136%|

See [the original repository](https://github.com/shicai/DenseNet-Caffe).

## Input

The model input is a blob that consists of a single image of `1, 3, 224, 224` in `BGR`
order. Before passing the image blob into the network, subtract BGR mean values
as follows: [103.94, 116.78, 123.68]. In addition, values must be divided by 0.017.

### Original Model

Image, name - `data`,  shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [103.94, 116.78, 123.68], scale value - 58.8235294117647

### Converted Model

Image, name - `data`,  shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

The model output for `densenet-121` is a typical object classifier output for 1000 different
classifications matching those in the ImageNet database.

### Original Model

Object classifier according to ImageNet classes, name - `fc6`,  shape - `1, 1000, 1, 1`, contains predicted
probability for each class in logits format.

### Converted Model

Object classifier according to ImageNet classes, name - `fc6`,  shape - `1, 1000, 1, 1`, contains predicted
probability for each class in logits format.

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

* [Classification Benchmark C++ Demo](../../../demos/classification_benchmark_demo/cpp/README.md)
* [Classification Python\* Demo](../../../demos/classification_demo/python/README.md)

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/liuzhuang13/DenseNet/master/LICENSE):

```
Copyright (c) 2016, Zhuang Liu.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name DenseNet nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
