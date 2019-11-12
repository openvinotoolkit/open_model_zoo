# human-pose-estimation-3d-0001

## Use Case and High-Level Description

Multi-person 3D human pose estimation model based on [Lightweight OpenPose](https://arxiv.org/pdf/1811.12004.pdf) and [Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB](https://arxiv.org/pdf/1712.03453.pdf) papers.

## Example

![](./human-pose-estimation-3d-0001.jpg)

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| MPJPE (mm)                                                    | 100.45                  |
| GFlops                                                        | 18.998                  |
| MParams                                                       | 5.074                   |
| Source framework                                              | PyTorch\*               |

## Performance

## Inputs

1. name: "data" , shape: [1x3x256x448] - An input image in the format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

## Outputs

1. The net outputs three blobs with shapes: [1, 57, 32, 56], [1, 19, 32, 56], and [1, 38, 32, 56]. The first blob contains coordinates in 3D space, the second one contains keypoint heatmaps and the third is keypoint pairwise relations (part affinity fields).

## Legal Information
[LICENSE](https://raw.githubusercontent.com/opencv/openvino_training_extensions/develop/LICENSE)

[*] Other names and brands may be claimed as the property of others.
