# densenet-121-tf

## Use Case and High-Level Description

This is an Tensorflow version of `densenet-121` model, one of the DenseNet
group of models designed to perform image classification. The weights were converted from DenseNet-Keras Models. For details see [repository](https://github.com/pudae/tensorflow-densenet/), [paper](https://arxiv.org/pdf/1608.06993.pdf)

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 5.289                                     |
| MParams                         | 7.971                                     |
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
   Mean values - [123.68, 116.78, 103.94], scale factor for each channel - 58.8235294

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

1. Name: `densenet121/predictions/Reshape_1`, contains floating point values in range [0, 1], which represent probabilities for classes in dataset.

### Converted model

1. Names: `densenet121/predictions/Reshape_1/Transpose`, shape: [1, 1, 1, 1000], contains floating point values in range [0, 1], which represent probabilities for classes in dataset.

## Legal Information
[https://raw.githubusercontent.com/pudae/tensorflow-densenet/master/LICENSE]()