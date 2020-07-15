# human-pose-estimation-3d-0001

## Use Case and High-Level Description

Multi-person 3D human pose estimation model based on the [Lightweight OpenPose](https://arxiv.org/abs/1811.12004) and [Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB](https://arxiv.org/abs/1712.03453) papers.

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

Name: `data`, shape: `[1x3x256x448]`. An input image in the `[BxCxHxW]` format,
where:

- B - batch size
- C - number of channels
- H - image height
- W - image width

Expected color order is BGR.

## Outputs

The net outputs three blobs with the following shapes: `[1, 57, 32, 56]`, `[1, 19, 32, 56]`, and `[1, 38, 32, 56]`. The first blob contains coordinates in 3D space, the second blob contains keypoint heatmaps and the third blob is keypoint pairwise relations (part affinity fields).

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/opencv/openvino_training_extensions/develop/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../../licenses/APACHE-2.0.txt).

[*] Other names and brands may be claimed as the property of others.
