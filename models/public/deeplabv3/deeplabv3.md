# deeplab-v3

## Use Case and High-Level Description

DeepLab is a state-of-art deep learning model for semantic image segmentation. For details see [paper](https://arxiv.org/pdf/1706.05587.pdf)

## Example

## Specification

| Metric            | Value                |
|-------------------|----------------------|
| Type              | Semantic segmentation|
| GFLOPs            | 11.469               |
| MParams           | 23.819               |
| Source framework  | Tensorflow           |

## Accuracy

## Performance

## Input

### Original model

1. Name: `ImageTensor`, shape: [1x513x513x3] - An input image in the format [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order - BGR.

### Converted model

1. Name: `mul_1/placeholder_port_1`, shape: [1x3x513x513] - An input image in the format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

## Output

### Original model

1. Name: `ArgMax`, shape: [1x513x513] in [BxHxW] format, where

    - B - batch size
    - H - image height
    - W - image width

Contains integer values in range [0, 20], which represents the index of predicted class for each image pixel.

### Converted model

1. Name: `ArgMax/Squeeze`, shape: [1x513x513] in [BxHxW] format, where

    - B - batch size
    - H - image height
    - W - image width

Contains integer values in range [0, 20], which represents the index of predicted class for each image pixel.

## Legal Information
[https://raw.githubusercontent.com/tensorflow/models/master/LICENSE]()