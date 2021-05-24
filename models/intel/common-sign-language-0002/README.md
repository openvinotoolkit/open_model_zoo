# common-sign-language-0002

## Use Case and High-Level Description

A human gesture recognition model for the Common-Sign-Language gesture recognition scenario.
The model support 12 common single-hand gestures:
* Digits: 0, 1, 2, 3, 4, 5
* Sliding Two Fingers Up / Down / Left / Right
* Thumb Up / Down

The model uses an S3D framework with MobileNet V3 backbone and accepts a stack of
frames (8 frames) sampled with a constant frame rate (15 FPS) and produces a prediction
on the input clip.

## Specification

| Metric                                  | Value        |
|-----------------------------------------|--------------|
| Top-1 accuracy (continuous CSL)         | 98.00%       |
| GFlops                                  | 4.2269       |
| MParams                                 | 4.1128       |
| Source framework                        | PyTorch\*    |

## Inputs

Image sequence, name: `input`, shape: `1, 3, 8, 224, 224` in the format `B, C, T, H, W`, where:

 - `B` - batch size
 - `C` - number of channels
 - `T` - duration of input clip
 - `H` - image height
 - `W` - image width

## Outputs

The model outputs a tensor with the shape `1, 12` in the format `B, L`, where:

- `B` - batch size
- `L` - logits vector for each performed CSL gestures

## Legal Information
[\*] Other names and brands may be claimed as the property of others.
