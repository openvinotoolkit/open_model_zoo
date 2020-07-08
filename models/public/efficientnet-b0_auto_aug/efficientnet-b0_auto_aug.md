# efficientnet-b0_auto_aug

## Use Case and High-Level Description

The `efficientnet-b0_auto_aug` model is one of the [EfficientNet](https://arxiv.org/abs/1905.11946)
models designed to perform image classification, trained with
[AutoAugmentation preprocessing](https://arxiv.org/abs/1805.09501).
This model was pretrained in TensorFlow\*.
All the EfficientNet models have been pretrained on the ImageNet\* image database.
For details about this family of models, check out the [TensorFlow Cloud TPU repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.819         |
| MParams           | 5.268         |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 76.43%          | 76.43%           |
| Top 5  | 93.04%          | 93.04%           |

## Performance

## Input

### Original Model

Image, name - `image`,  shape - `[1x224x224x3]`, format is `[BxHxWxC]`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.

### Converted Model

Image, name - `sub/placeholder_port_0`,  shape - `[1x3x224x224]`, format is `[BxCxHxW]`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `logits`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in the [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name - `efficientnet-b0/model/head/dense/MatMul`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in the [0, 1] range

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/tpu/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-TPU.txt](../licenses/APACHE-2.0-TF-TPU.txt).
