# mobilenet-v2-1.0-224

## Use Case and High-Level Description

`mobilenet-v2-1.0-224` is one of MobileNet\* models, which are small, low-latency, low-power, and parameterized to meet the resource constraints of a variety of use cases. They can be used for classification, detection, embeddings, and segmentation like other popular large-scale models. For details, see the [paper](https://arxiv.org/abs/1704.04861).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 0.615                                     |
| MParams                         | 3.489                                     |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 71.85%|
| Top 5  | 90.69%|

## Performance

## Input

### Original Model

Image, name: `input` , shape: [1x224x224x3], format: [BxHxWxC], where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.
   Mean values: [127.5, 127.5, 127.5], scale factor for each channel: 127.5.

### Converted Model

Image, name: `input`, shape: [1x3x224x224], format: [BxCxHxW], where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: BGR.

## Output

### Original Model

Name: `MobilenetV2/Predictions/Reshape_1`.
Probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format.

### Converted Model

Name: `MobilenetV2/Predictions/Softmax`.
Probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format.
Shape: [1,1001], format: [BxC],
    where:

    - B - batch size
    - C - vector of probabilities.

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).
