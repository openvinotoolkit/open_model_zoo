# mobilenet-v2-pytorch

## Use Case and High-Level Description

MobileNet V2 is image classification model pretrained on ImageNet dataset. This
is a PyTorch implementation of MobileNetV2 architecture as described in
the paper ["Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification,
Detection and Segmentation"](https://arxiv.org/abs/1801.04381).

The model input is a blob that consists of a single image of "1x3x224x224"
in RGB order.

The model output is typical object classifier for the 1000 different classifications
matching with those in the ImageNet database.

## Example

See [here](https://github.com/tonylins/pytorch-mobilenet-v2)

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.615         |
| MParams           | 3.489         |
| Source framework  | PyTorch\*     |

## Accuracy

| Metric | Original model | Converted model |
|--------|----------------|-----------------|
| Top 1  | 71.8%          | 71.8%           |
| Top 5  | 90.396%          | 90.396%       |

## Performance

## Input

### Original model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.
Mean values - [123.675,116.28,103.53], scale value - [58.624,57.12,57.375]

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/tonylins/pytorch-mobilenet-v2/master/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).
