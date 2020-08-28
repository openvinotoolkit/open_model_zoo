# mobilenet-v1-0.25-128

## Use Case and High-Level Description

`mobilenet-v1-0.25-128` is one of MobileNets - small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models are used. For details, see [paper](https://arxiv.org/abs/1704.04861).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 0.028                                     |
| MParams                         | 0.468                                     |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 40.54%|
| Top 5  | 65%   |

## Performance

## Input

### Original Model

Image, name: `input` , shape: [1x128x128x3], format: [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.
   Mean values: [127.5, 127.5, 127.5], scale factor for each channel: 127.5

### Converted Model

Image, name: `input` , shape: [1x3x128x128], format: [BxCxHxW],
where:

   - B - batch size
   - C - number of channels
   - H - image height
   - W - image width

Expected color order: BGR.

## Output

### Original Model

Probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format. Name: `MobilenetV1/Predictions/Reshape_1`.

### Converted Model

Probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format. Name: `MobilenetV1/Predictions/Softmax`, shape: [1,1001],  format: [BxC],
    where:

    - B - batch size
    - C - vector of probabilities.

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).
