# asl-recognition-0004

## Use Case and High-Level Description

A human gesture recognition model for the American Sign Language (ASL) recognition scenario
(word-level recognition). The model uses an S3D framework with MobileNet V3 backbone. Please refer
to the [MS-ASL-100](https://www.microsoft.com/en-us/research/project/ms-asl/) dataset specification
to see the list of gestures that are recognized by this model.

The model accepts a stack of frames sampled with a constant frame rate (15 FPS) and produces a prediction
on the input clip.

## Example

![](./assets/asl-recognition-0004.jpg)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Top-1 accuracy (MS-ASL-100)     | 0.847                                     |
| GFlops                          | 6.660                                     |
| MParams                         | 4.133                                     |
| Source framework                | PyTorch\*                                 |

## Inputs

Image sequence, name: `input`, shape: `1, 3, 16, 224, 224` in the format `B, C, T, H, W`, where:

 - `B` - batch size
 - `C` - number of channels
 - `T` - duration of input clip
 - `H` - image height
 - `W` - image width

## Outputs

The model outputs a tensor with the shape `1, 100` in the format `B, L`, where:

- `B` - batch size
- `L` - logits vector for each performed ASL gestures

## Legal Information
[\*] Other names and brands may be claimed as the property of others.
