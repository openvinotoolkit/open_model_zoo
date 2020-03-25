# head-pose-estimation-adas-0001

## Use Case and High-Level Description

Head pose estimation network based on simple, handmade CNN architecture. Angle regression
layers are convolutions + ReLU + batch norm + fully connected with
one output.

## Validation Dataset

[Biwi Kinect Head Pose Database](https://icu.ee.ethz.ch/research/datsets.html)

## Example

![](./head-pose-estimation-adas-0001.png)

## Specification

| Metric                | Value                                       |
|-----------------------|---------------------------------------------|
| Supported ranges      | YAW [-90,90], PITCH [-70,70], ROLL [-70,70] |
| GFlops                | 0.105                                       |
| MParams               | 1.911                                       |
| Source framework      | Caffe*                                      |

## Accuracy

| Angle |  [Mean](https://en.wikipedia.org/wiki/Mean_absolute_error) ± [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) of absolute error |
|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| yaw   |  5.4 ± 4.4                                                                                                                                            |
| pitch |  5.5 ± 5.3                                                                                                                                            |
| roll  |  4.6 ± 5.6                                                                                                                                            |

## Performance

## Inputs

1. name: "data" , shape: [1x3x60x60] - An input image in [1xCxHxW] format. Expected color order is BGR.

## Outputs

Output layer names in Inference Engine format:

1. name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
2. name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
3. name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).

Output layer names in Caffe* format:

1. name: "fc_y", shape: [1, 1] - Estimated yaw (in degrees).
2. name: "fc_p", shape: [1, 1] - Estimated pitch (in degrees).
3. name: "fc_r", shape: [1, 1] - Estimated roll (in degrees).


Each output contains one float value that represents value in Tait-Bryan angles
(yaw, pitch or roll).

## Legal Information
[*] Other names and brands may be claimed as the property of others.
