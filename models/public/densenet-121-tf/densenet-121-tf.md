# densenet-121-tf

## Use Case and High-Level Description

This is an Tensorflow\* version of `densenet-121` model, one of the DenseNet\*
group of models designed to perform image classification. The weights were converted from DenseNet-Keras Models. For details, see [repository](https://github.com/pudae/tensorflow-densenet/) and [paper](https://arxiv.org/pdf/1608.06993.pdf).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 5.289                                     |
| MParams                         | 7.971                                     |
| Source framework                | Tensorflow\*                              |

## Performance

## Input

### Original Model

Image, name: `Placeholder` , shape: [1x224x224x3], format: [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.
   Mean values: [123.68, 116.78, 103.94], scale factor for each channel: 58.8235294

### Converted Model

Image, name: `Placeholder`, shape: [1x3x224x224], [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: BGR.

## Output

### Original Model

Floating point values in a range [0, 1], which represent probabilities for classes in a dataset. Name: `densenet121/predictions/Reshape_1`.

### Converted Model

Floating point values in a range [0, 1], which represent probabilities for classes in a dataset. Name: `densenet121/predictions/Reshape_1/Transpose`, shape - [1, 1, 1, 1000].

## Legal Information

[https://raw.githubusercontent.com/pudae/tensorflow-densenet/master/LICENSE]()
