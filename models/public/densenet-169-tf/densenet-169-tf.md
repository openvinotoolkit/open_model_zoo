# densenet-169-tf

## Use Case and High-Level Description

This is a TensorFlow\* version of `densenet-169` model, one of the DenseNet
group of models designed to perform image classification. The weights were converted from DenseNet-Keras Models. For details, see [repository](https://github.com/pudae/tensorflow-densenet/) and [paper](https://arxiv.org/abs/1608.06993).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 6.16                                      |
| MParams                         | 14.139                                    |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 75.76% |
| Top 5  | 92.81%|

## Performance

## Input

### Original Model

Name: `Placeholder` , shape: [1x224x224x3]. An input image is the [BxHxWxC] format,
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.
   Mean values: [123.68, 116.78, 103.94], scale factor for each channel: 58.8235294

### Converted Model

Name: `Placeholder`, shape: [1x3x224x224]. An input image is in the [BxCxHxW] format,
where:

   - B - batch size
   - C - number of channels
   - H - image height
   - W - image width

Expected color order: BGR.

## Output

### Original Model

Name:`densenet169/predictions/Reshape_1`. Contains floating point values in a range [0, 1], which represent probabilities for classes in a dataset.

### Converted Model

Name: `densenet169/predictions/Reshape_1/Transpose`, shape: [1, 1, 1, 1000]. Contains floating point values in a range [0, 1], which represent probabilities for classes in a dataset.

## Legal Information
The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/pudae/tensorflow-densenet/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-DenseNet.txt](../licenses/APACHE-2.0-TF-DenseNet.txt).
