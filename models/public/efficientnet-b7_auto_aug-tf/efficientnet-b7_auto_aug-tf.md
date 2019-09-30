# efficientnet-b7-tf

## Use Case and High-Level Description

The `efficientnet-b7_auto_aug-tf` model is one of the [EfficientNet](https://arxiv.org/abs/1905.11946)
group of models designed to perform image classification, trained with
[AutoAugmentation preprocessing](https://arxiv.org/abs/1805.09501).
This model was pretrained in TensorFlow\*.
All the EfficientNet models have been pretrained on the ImageNet image database.
For details about this family of models, check out the [repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 77.618        |
| MParams           | 66.193        |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 84.68          | 84.68           |
| Top 5  | 97.09          | 97.09           | 

## Performance

## Input

### Original model

Image, name - `image`,  shape - `[1x600x600x3]`, format is `[BxHxWxC]` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.

### Converted model

Image, name - `sub/placeholder_port_0`,  shape - `[1x600x600x3]`, format is `[BxHxWxC]` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `logits`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `efficientnet-b7/model/head/dense/MatMul`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Legal Information

[LICENSE](https://raw.githubusercontent.com/tensorflow/tpu/master/LICENSE)