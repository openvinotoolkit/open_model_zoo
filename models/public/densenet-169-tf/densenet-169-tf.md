# densenet-169-tf

## Use Case and High-Level Description

This is an Tensorflow implementation of DenseNet by G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten with ImageNet pretrained models. The weights are converted from DenseNet-Keras Models. For details see [repository](https://github.com/pudae/tensorflow-densenet/), [paper](https://arxiv.org/pdf/1608.06993.pdf)

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 6.16                                      |
| MParams                         | 14.139                                    |
| Source framework                | Tensorflow                                |

## Performance

## Input

### Original model

1. Name: `Placeholder` , shape: [1x224x224x3] - An input image in the format [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order - BGR.

### Converted model

1. Name: `Placeholder`, shape: [1x3x224x224] - An input image in the format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

## Output

### Original model

1. Name: `densenet169/predictions/Reshape_1`, contains floating point values in range [0, 1], which represent probabilities for classes in dataset.

### Converted model

1. Names: `densenet169/predictions/Reshape_1/Transpose`, shape: [1, 1, 1, 1000], contains floating point values in range [0, 1], which represent probabilities for classes in dataset.

## Legal Information
[https://raw.githubusercontent.com/pudae/tensorflow-densenet/master/LICENSE]()