# efficientnet-b7-pt

## Use Case and High-Level Description

The `efficientnet-b7-pt` model is one of the [EfficientNet](https://arxiv.org/abs/1905.11946) group of models designed to perform image classification. This model was pretrained in TensorFlow\*, then weights were converted to PyTorch\*. All the EfficientNet models have been pretrained on the ImageNet image database. For details about this family of models, check out the [repository](https://github.com/rwightman/gen-efficientnet-pytorch).

The model input is a blob that consists of a single image 3x600x600 in RGB
order. The RGB mean values need to be subtracted as follows: [123.675,116.28,103.53]
before passing the image blob into the network. In addition, values must be divided
by [58.395,57.12,57.375].

The model output for `efficientnet-b7-pt` is the typical object classifier output for
the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 77.618        |
| MParams           | 66.193        |
| Source framework  | PyTorch\*     |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 84.42%         | 84.42%          |
| Top 5  | 96.91%         | 96.91%          | 

## Performance

## Input

### Original model

Image, name - `data`,  shape - `1,3,600,600`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.
Mean values - [123.675,116.28,103.53], scale values - [58.395,57.12,57.375].

### Converted model

Image, name - `data`,  shape - `1,3,600,600`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Legal Information

[LICENSE](https://raw.githubusercontent.com/rwightman/gen-efficientnet-pytorch/a36e2b2cd1bd122a508a6fffeaa7606890f8c882/LICENSE)
