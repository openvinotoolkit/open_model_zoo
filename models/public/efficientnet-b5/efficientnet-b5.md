# efficientnet-b5

## Use Case and High-Level Description

The `efficientnet-b5` model is one of the [EfficientNet](https://arxiv.org/abs/1905.11946)
models designed to perform image classification.
This model was pretrained in TensorFlow\*.
All the EfficientNet models have been pretrained on the ImageNet\* image database.
For details about this family of models, check out the [TensorFlow Cloud TPU repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 21.252        |
| MParams           | 30.303        |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 83.33%          | 83.33%           |
| Top 5  | 96.67%          | 96.67%           |

## Performance

## Input

### Original Model

Image, name - `image`,  shape - `[1x456x456x3]`, format is `[BxHxWxC]` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.

### Converted Model

Image, name - `sub/placeholder_port_0`,  shape - `[1x456x456x3]`, format is `[BxHxWxC]` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `logits`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in the [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name - `efficientnet-b5/model/head/dense/MatMul`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in the [0, 1] range

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/tpu/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-TPU.txt](../licenses/APACHE-2.0-TF-TPU.txt).
