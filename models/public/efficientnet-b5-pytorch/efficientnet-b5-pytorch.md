# efficientnet-b5-pytorch

## Use Case and High-Level Description

The `efficientnet-b5-pytorch` model is one of the [EfficientNet](https://arxiv.org/abs/1905.11946)
models designed to perform image classification. This model was pretrained in TensorFlow\*, then weights were converted to PyTorch\*. All the EfficientNet models have been pretrained on the ImageNet\* image database. For details about this family of models, check out the [EfficientNets for PyTorch repository](https://github.com/rwightman/gen-efficientnet-pytorch).


The model input is a blob that consists of a single image with the [3x456x456] shape in the RGB
order. Before passing the image blob to the network, do the following:
1. Subtract the RGB mean values as follows: [123.675,116.28,103.53]
2. Divide the RGB mean values by  [58.395,57.12,57.375]

The model output for `efficientnet-b5-pytorch` is the typical object classifier output for
the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 21.252        |
| MParams           | 30.303        |
| Source framework  | PyTorch\*     |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 83.69%          | 83.69%           |
| Top 5  | 96.71%          | 96.71%           |

## Performance

## Input

### Original Model

Image, name - `data`,  shape - `1,3,456,456`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.
Mean values - [123.675,116.28,103.53], scale values - [58.395,57.12,57.375].

### Converted Model

Image, name - `data`,  shape - `1,3,456,456`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in the [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in the [0, 1] range

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/rwightman/gen-efficientnet-pytorch/5e91628ed98250989a7ddd20abfe27385e0493c1/LICENSE).
A copy of the license is provided in [APACHE-2.0-PyTorch-EfficientNet.txt](../licenses/APACHE-2.0-PyTorch-EfficientNet.txt).
