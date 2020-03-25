# deeplabv3

## Use Case and High-Level Description

DeepLab is a state-of-art deep learning model for semantic image segmentation. For details see [paper](https://arxiv.org/abs/1706.05587).

## Example

## Specification

| Metric            | Value                |
|-------------------|----------------------|
| Type              | Semantic segmentation|
| GFLOPs            | 11.469               |
| MParams           | 23.819               |
| Source framework  | TensorFlow\*         |

## Accuracy

## Performance

## Input

### Original model

Image, name: `ImageTensor`, shape: [1x513x513x3], format: [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.

### Converted Model

Image, name: `mul_1/placeholder_port_1`, shape: [1x3x513x513], format: [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: RGB.

> **NOTE**: By default, Open Model Zoo demos expect input with BGR channels order. Reverse input channels operation via Model Optimizer conversion can not be applied to this model in proper way. For running this model with Open Model Zoo demos, you need to manually rearrange the default channels order in the demo application.

## Output

### Original Model

Integer values in a range [0, 20], which represent an index of a predicted class for each image pixel. Name: `ArgMax`, shape: [1x513x513] in [BxHxW] format, where

    - B - batch size
    - H - image height
    - W - image width


### Converted Model

Integer values in a range [0, 20], which represent an index of a predicted class for each image pixel. Name: `ArgMax/Squeeze`, shape: [1x513x513] in [BxHxW] format, where

    - B - batch size
    - H - image height
    - W - image width


## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).
