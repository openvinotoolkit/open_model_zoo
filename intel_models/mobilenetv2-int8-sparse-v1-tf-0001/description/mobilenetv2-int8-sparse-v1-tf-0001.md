# mobilenetv2-int8-sparse-v1-tf-0001

## Use Case and High-Level Description

This is the MobileNet v2 model that is designed to perform image classification. The model has been pretrained on the ImageNet image database and then pruned to **30.8%** of sparsity and quantized to INT8 fixed-point precision using so-called Quantization-aware training approach implemented in TensorFlow framework. The sparsity is represented by zeros inside the weights of Convolutional and Fully-conneted layers. For details about the original floating point model, check out the [paper](https://arxiv.org/pdf/1801.04381.pdf).

The model input is a blob that consists of a single image of "1x224x224x3" in BGR order.

The model output for `mobilenetv2-int8-sparse-v1-tf-0001` is the usual object classifier output for the 1001 different classifications matching those in the ImageNet database (the first item represents the background).

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.615         |
| MParams           | 3.489         |
| Source framework  | TensorFlow    |

## Accuracy

The quality metrics calculated on ImageNet validation dataset is 71.42% accuracy top-1.

| Metric                    | Value         |
|---------------------------|---------------|
| Accuracy top-1 (ImageNet) |        71.42% |

## Performance

## Input

Image, shape - `1,224,224,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`

## Output

Object classifier according to ImageNet classes, shape -`1,1001`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

