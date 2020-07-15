# se-resnet-101

## Use Case and High-Level Description

[ResNet-101 with Squeeze-and-Excitation blocks](https://arxiv.org/abs/1709.01507)

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 15.239        |
| MParams           | 49.274        |
| Source framework  | Caffe\*       |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 78.252%|
| Top 5  | 94.206%|

## Performance

## Input

### Original model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [104.0,117.0,123.0].

### Converted model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/hujie-frank/SENet/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-SENet.txt](../licenses/APACHE-2.0-SENet.txt).
