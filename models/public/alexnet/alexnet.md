# alexnet

## Use Case and High-Level Description

The `alexnet` model is designed to perform image classification. Just like other common classification models, the `alexnet` model has been pretrained on the ImageNet image database. For details about this model, check out the [paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

The model input is a blob that consists of a single image of "1x3x227x227" in BGR order.

The model output for `alexnet` is the usual object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 1.5           |
| MParams           | 60.965        |
| Source framework  | Caffe         |

## Accuracy

## Performance

## Input

Name - `data`, shape - `1,3,227,227`, image format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

Name: `prob`

## Legal Information

[https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/readme.md]()
