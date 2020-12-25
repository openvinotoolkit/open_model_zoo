# anti-spoof-mn3

## Use Case and High-Level Description

The `anti-spoof-mn3` model is an anti-spoofing binary classificator based on the [MobileNetv3](https://arxiv.org/abs/1905.02244), trained on the [CelebA-Spoof dataset](https://arxiv.org/abs/2007.12342). It's a small, light model, trained to predict whether or not a spoof RGB image given to the input. A lot of advanced techniques have been tried and selected the best suit options for the task.
For details see original [repository](https://github.com/kirillProkofiev/light-weight-face-anti-spoofing).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 0.15                                    |
| MParams                         | 3.02                                    |
| Source framework                | PyTorch\*                              |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| ACER   | 3.81%          | 3.81%           |

## Input

### Original Model

Image, name: `actual_input_1` , shape: [1x3x128x128], format: [BxCxHxW], where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: RGB.
   Mean values: [151.2405,119.5950,107.8395], scale factor: [63.0105,56.4570,55.0035]

### Converted Model

Image, name: `actual_input_1` , shape: [1x3x128x128], format: [BxCxHxW], where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: BGR.

## Output

### Original model

Probabilities for two classes (0 class is a real person, 1 - is a spoof image). Name: `output1` Shape: [1,2], format: [BxC],
    where:

    - B - batch size
    - C - vector of probabilities.

### Converted model

Probabilities for two classes (0 class is a real person, 1 - is a spoof image). Name: `output1` Shape: [1,2], format: [BxC],
    where:

    - B - batch size
    - C - vector of probabilities.

## Legal Information

The original model is distributed under the
[MIT License](https://raw.githubusercontent.com/kirillProkofiev/light-weight-face-anti-spoofing/master/LICENSE).
