# single-human-pose-estimation-0001

## Use Case and High-Level Description
Single human pose estimation model based on https://arxiv.org/abs/1906.04104.

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| AP(coco orig)                                                 | 69.04%                   |
| GFlops                                                        | 60.125                  |
| MParams                                                       | 33.165                  |
| Source framework                                              | PyTorch\*               |


## Inputs

### Original model

Name: "data" , shape: [1x3x384x288] - An input image in the format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - RGB. Mean values - [123.675,116.28,103.53]. Scale values - [58.395,57.12,57.375]

### Converted model

Name: "data" , shape: [1x3x384x288] - An input image in the format [BxCxHxW],
where:

   - B - batch size
   - C - number of channels
   - H - image height
   - W - image width

Expected color order: BGR.

## Outputs

### Original model

The net outputs list of tensor. Count of list elements is 6. Every tensor with shapes: [1x17x48x36] ( For every keypoint own heatmap). The six outputs are necessary in order to calculate the loss in during training. But in the future, for obtaining the results of prediction and postprocessing them, the last output is used. Each following tensor gives more accurate predictions ( in context metric AP).


### Converted model

The net outputs tensor with shapes: [1x17x48x36]. ( For every keypoint own heatmap)

## Legal Information
The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/opencv/openvino_training_extensions/develop/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../../licenses/APACHE-2.0.txt).

[*] Other names and brands may be claimed as the property of others.
