# inceptionv3-int8-tf-0001

## Use Case and High-Level Description

This is the Inception v3 model that is designed to perform image classification. The model has been pretrained on the ImageNet image database and then quantized to INT8 fixed-point precision using so-called Quantization-aware training approach implemented in TensorFlow framework. For details about the original floating point model, check out the [paper](https://arxiv.org/pdf/1512.03385.pdf).

The model input is a blob that consists of a single image of "1x299x299x3" in BGR order.

The model output for `inceptionv3-int8-tf-0001` is the usual object classifier output for the 1001 different classifications matching those in the ImageNet database (the first item represents the background).

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 11.469        |
| MParams           | 23.819        |
| Source framework  | TensorFlow    |

## Accuracy

The quality metrics calculated on ImageNet validation dataset is 78.07% accuracy top-1.

| Metric                    | Value         |
|---------------------------|---------------|
| Accuracy top-1 (ImageNet) |        78.07% |

## Performance

## Input

Image, shape - `1,299,299,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`

## Output

Object classifier according to ImageNet classes, shape -`1,1001`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

