# hbonet-1.0

## Use Case and High-Level Description

The `hbonet-1.0` model is one of the classification models from https://github.com/d-li14/HBONet with `width_mult=1.0`

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.305         |
| MParams           |               |
| Source framework  | PyTorch\*     |

## Accuracy
Top-1: 73.1%
Top-5: 91.0%

## Performance

## Input

### Original Model

Image, name: `input`, shape: [1x224x224x3], format: [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.
   Mean values: [123.675, 116.28, 103.53], scale factor for each channel: [58.395, 57.12, 57.375] 

### Converted Model

Image, name: `input`, shape: [1x3x224x224], format: [BxCxHxW],
where:

   - B - batch size
   - C - number of channels
   - H - image height
   - W - image width

Expected color order: BGR.

## Output

Object classifier according to ImageNet classes, shape: [1,1000] in [BxC] format, where:

    - B - batch size
    - C - vector of probabilities for all dataset classes.

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/d-li14/HBONet/master/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).
