# pspnet-pytorch

## Use Case and High-Level Description

pspnet-pytorch is a semantic segmentation model, pretrained on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset for 21 object classes. The model was built on [ResNetV1-50](https://arxiv.org/pdf/1812.01187.pdf) backbone and PSP segmentation head. This model is used for pixel-level prediction tasks. For details see [repository](https://github.com/open-mmlab/mmsegmentation/tree/master).

## Specification

| Metric            | Value                |
|-------------------|----------------------|
| Type              | Semantic segmentation|
| GFlops            | 357.1719             |
| MParams           | 46.5827              |
| Source framework  | PyTorch\*            |

## Accuracy

| Metric | Value |
| ------ | ----- |
| mean_iou  | 70.6%|

Accuracy metrics were obtained with fixed input resolution 512x512.

## Input

### Original model

Image, name: `input.1`, shape: `1, 3, 512, 512`, format: `B, C, H, W`,
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: RGB.
Mean values: [123.675, 116.28, 103.53], scale values: [58.395, 57.12, 57.375]

### Converted Model

Image, name: `input.1`, shape: `1, 3, 512, 512`, format: `B, C, H, W`,
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: BGR.

## Output

### Original Model

Integer values in a range [0, 20], which represent an index of a predicted class for each image pixel. Name: `Unsqueeze_259`, shape: `1, 1, 512, 512` in `B, 1, H, W` format, where

    - B - batch size
    - H - image height
    - W - image width

### Converted Model

1. Integer values in a range [0, 20], which represent an index of a predicted class for each image pixel. Name: `Unsqueeze_259`, shape: `1, 1, 512, 512` in `B, 1, H, W` format, where

    - B - batch size
    - H - image height
    - W - image width

2. Float values, which represent scores of a predicted class for each image pixel. Name: `8802.0`, shape: `1, 1, 512, 512` in `B, 1, H, W` format, where

    - B - batch size
    - H - image height
    - W - image width

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/open-mmlab/mmsegmentation/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-MMSegmentation-Models.txt](../licenses/APACHE-2.0-MMSegmentation-Models.txt).
