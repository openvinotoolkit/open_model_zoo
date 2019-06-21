# inception-resnet-v2

## Use Case and High-Level Description

The `inception-resnet-v2` model is one of the [Inception](https://arxiv.org/pdf/1602.07261.pdf) family of models designed to perform image classification.1 Like the other Inception models, the `inception-resnet-v2` model has been pretrained on the ImageNet image database. For details about this family of models, check out the paper.

The model input is a blob that consists of a single image of "1x3x299x299" in BGR order.

The model output for `inception-resnet-v2` is the typical object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 26.405        |
| MParams           | 55.813        |
| Source framework  | Caffe         |

## Accuracy

## Performance

## Input

Image, shape - `1,3,299,299`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

Object classifier according to ImageNet classes, shape-`1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

## Legal Information

[https://raw.githubusercontent.com/soeaver/caffe-model/master/LICENSE]()
