# mixnet-l

## Use Case and High-Level Description
MixNets are a family of mobile-sizes image classification models equipped with MixConv,
a new type of mixed depthwise convolutions. The `mixnet-l` model is one of the
[MixNet](https://arxiv.org/abs/1907.09595) models designed.
This model was pretrained in TensorFlow\*.
All the MixNet models have been pretrained on the ImageNet* image database.
For details about this family of models, check out the [TensorFlow Cloud TPU repository](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet).

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.565        |
| MParams           | 7.3        |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 78.3%          | ? %           |
| Top 5  | 93.91%          | ? %           |

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

Image, name - `IteratorGetNext/placeholder_out_port_0`,  shape - `[1x3x224x224]`, format is `[BxCxHxW]`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `logits`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name - `mixnet-l/mixnet_model/head/dense/BiasAdd/Add`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/tpu/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-TPU.txt](../licenses/APACHE-2.0-TF-TPU.txt).