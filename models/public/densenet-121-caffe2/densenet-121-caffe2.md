# densenet-121-caffe2

## Use Case and High-Level Description

This is a Caffe2\* version of `densenet-121` model, one of the DenseNet
group of models designed to perform image classification. This model
was converted from Caffe\* to Caffe2\* format.
For details see repository <https://github.com/caffe2/models/tree/master/densenet121>,
paper <https://arxiv.org/abs/1608.06993>.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 5.723         |
| MParams           | 7.971         |
| Source framework  | Caffe2\*      |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 74.904% |
| Top 5  | 92.192% |

## Performance

## Input

### Original model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [103.94,116.78,123.68], scale value - 58.8235294.

### Converted model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `fc6`,  shape - `1,1000,1,1`, contains predicted
probability for each class in logits format.

### Converted model

Object classifier according to ImageNet classes, name - `fc6`,  shape - `1,1000,1,1`, contains predicted
probability for each class in logits format.

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/caffe2/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).
