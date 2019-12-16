# asl-recognition-0003

## Use Case and High-Level Description

This is an human gesture recognition model for American Sign Language (ASL) recognition scenario (word-level recognition). The model uses S3D framework with MobileNet V3 backbone. Please refer to the [MS-ASL-100\*](https://www.microsoft.com/en-us/research/project/ms-asl/) dataset specification to see list of gestures that are recognised by this model.

This model accepts stack of frames sampled with constant framerate (15 FPS) and produces prediction on the input clip.

## Example

![](./asl-recognition-0003.jpg)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Top-1 accuracy (MS-ASL-100\*)   | 0.836                                     |
| GFlops                          | 6.651                                     |
| MParams                         | 4.129                                     |
| Source framework                | PyTorch\*                                 |


## Performance

## Inputs

Name: "input" , shape: [1x3x16x224x224] - An input image sequence in the format [BxCxTxHxW], where:
 - B - Batch size.
 - C - Number of channels.
 - T - Duration of input clip.
 - H - Image height.
 - W - Image width.

## Outputs

The model outputs a tensor with the shape [Bx100], each row is a logits vector of performed ASL gestures.

## Legal Information
[\*] Other names and brands may be claimed as the property of others.
