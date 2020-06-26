# inception-resnet-v2-tf

## Use Case and High-Level Description

The `inception-resnet-v2` model is one of the Inception family of models designed to perform image classification. For details about this family of models, check out the [paper](https://arxiv.org/abs/1602.07261).

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 22.227                                    |
| MParams                         | 30.223                                    |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 80.14% |
| Top 5  | 95.10% |

## Performance

## Input

### Original Model

Image, name: `input` , shape: [1x299x299x3], format: [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.
   Mean values: [127.5, 127.5, 127.5], scale factor for each channel: 127.5

### Converted Model

Image, name: `input`, shape: [1x3x299x299], format: [BxCxHxW],
where:

   - B - batch size
   - C - number of channels
   - H - image height
   - W - image width

Expected color order: BGR.

## Output

### Original Model

Probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format. Name: `InceptionResnetV2/AuxLogits/Logits/BiasAdd`.

### Converted Model

Probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format. Name: `InceptionResnetV2/AuxLogits/Logits/MatMul`, shape: [1,1001] in [BxC] format,
    where:

    - B - batch size
    - C - vector of probabilities.

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).
