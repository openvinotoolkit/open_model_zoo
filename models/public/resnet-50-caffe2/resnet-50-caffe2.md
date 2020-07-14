# resnet-50-caffe2

## Use Case and High-Level Description

This is a Caffe2\* version of `resnet-50` model, designed to perform image classification.
This model was converted from Caffe\* to Caffe2\* format.
For details see repository <https://github.com/caffe2/models/tree/master/resnet50>,
paper <https://arxiv.org/abs/1512.03385>.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 8.216         |
| MParams           | 25.53         |
| Source framework  | Caffe2\*      |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 76.38% |
| Top 5  | 93.188%|

## Performance

## Input

### Original model

Image, name - `gpu_0/data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [103.53,116.28,123.675], scale values - [57.375,57.12,58.395].

### Converted model

Image, name - `gpu_0/data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `gpu_0/softmax`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `gpu_0/softmax`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/caffe2/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).
