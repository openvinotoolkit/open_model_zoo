# single human-pose-estimation-0001

## Use Case and High-Level Description
Single human pose estimation model based on https://arxiv.org/pdf/1906.04104.pdf.

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| AP(coco orig)                                                 | 68                      |
| GFlops                                                        | 60.125                  |
| MParams                                                       | 33.165                  |
| Source framework                                              | PyTorch\*               |


## Inputs

1. name: "data" , shape: [1x3x384x288] - An input image in the format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - RGB.

## Outputs

1. The net outputs tensor with shapes: [1x17x48x36]. ( For everyone keypoint own heatmap)

## Legal Information
[LICENSE](https://raw.githubusercontent.com/opencv/openvino_training_extensions/develop/LICENSE)

[*] Other names and brands may be claimed as the property of others.
