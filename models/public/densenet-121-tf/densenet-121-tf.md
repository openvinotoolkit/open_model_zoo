# densenet-121-tf

## Use Case and High-Level Description

This is a TensorFlow\* version of `densenet-121` model, one of the DenseNet\*
group of models designed to perform image classification. The weights were converted from DenseNet-Keras Models. For details, see [repository](https://github.com/pudae/tensorflow-densenet/) and [paper](https://arxiv.org/abs/1608.06993).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 5.289                                     |
| MParams                         | 7.971                                     |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 74.29% |
| Top 5  | 91.98%|

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

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/pudae/tensorflow-densenet/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-DenseNet.txt](../licenses/APACHE-2.0-TF-DenseNet.txt).
