# squeezenet1.1-caffe2

## Use Case and High-Level Description

This is a Caffe2\* version of `squeezenet1.1` model, designed to perform image classification.
This model was converted from Caffe\* to Caffe2\* format.
For details see repository <https://github.com/caffe2/models/tree/master/squeezenet>,
paper <https://arxiv.org/abs/1602.07360>.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.784         |
| MParams           | 1.235         |
| Source framework  | Caffe2\*      |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 56.502%|
| Top 5  | 79.576%|

## Performance

## Input

### Original model

Image, name - `data`, shape - `1,3,227,227`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [103.96,116.78,123.68].

### Converted model

Image, name - `data`, shape - `1,3,227,227`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `softmaxout`, shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `softmaxout`, shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/caffe2/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).
