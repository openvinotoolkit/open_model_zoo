# mobilenet-v3-small-1.0-224-tf

## Use Case and High-Level Description

`mobilenet-v3-small-1.0-224-tf` is one of MobileNets V3 - next generation of MobileNets,
based on a combination of complementary search techniques as well as a novel architecture design.
`mobilenet-v3-small-1.0-224-tf` is targeted for low resource use cases.
For details see [paper](https://arxiv.org/abs/1905.02244).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 0.121                                     |
| MParams                         | 2.537                                     |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 67.36%          | 67.36%           |
| Top 5  | 87.45%          | 87.45%           |

## Performance

## Input

### Original Model

Image, name: `input` , shape: [1x224x224x3], format: [BxHxWxC], where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.
   Mean values: [127.5, 127.5, 127.5], scale factor for each channel: 127.5

### Converted Model

Image, name: `input` , shape: [1x3x224x224], format: [BxCxHxW], where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: BGR.

## Output

### Original Model

Probabilities for all dataset classes (0 class is background). Name: `MobilenetV3/Predictions/Softmax`,
shape: [1,1001], format: [BxC],
    where:

    - B - batch size
    - C - vector of probabilities.

### Converted Model

Probabilities for all dataset classes (0 class is background). Name: `MobilenetV3/Predictions/Softmax`,
shape: [1,1001], format: [BxC],
    where:

    - B - batch size
    - C - vector of probabilities.

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).
