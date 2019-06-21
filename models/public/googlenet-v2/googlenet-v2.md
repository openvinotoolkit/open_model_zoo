# googlenet-v2

## Use Case and High-Level Description

The `googlenet-v2` model is the second of the [Inception](https://arxiv.org/pdf/1602.07261.pdf) family of models designed to perform image classification. Like the other Inception models, the `googlenet-v2` model has been pretrained on the ImageNet image database. For details about this family of models, check out the paper.

The model input is a blob that consists of a single image of "1x3x224x224" in BGR order.

The model output for `googlenet-v2` is the typical object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 4.058         |
| MParams           | 11.185        |
| Source framework  | Caffe         |

## Accuracy

## Performance

## Input

Name - `data`, shape - `1,3,224,224`, image format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

Name: `prob`

## Legal Information

[https://raw.githubusercontent.com/BVLC/caffe/master/LICENSE]()
