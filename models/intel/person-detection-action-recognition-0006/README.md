# person-detection-action-recognition-0006

## Use Case and High-Level Description

This is an action detector for the Smart Classroom scenario. It is based on the RMNet backbone that includes depth-wise convolutions to reduce the amount of computations for the 3x3 convolution block. The first SSD head from 1/8 and 1/16 scale feature maps has four clustered prior boxes and outputs detected persons (two class detector). The second SSD-based head predicts actions of the detected persons. Possible actions: sitting, writing, raising hand, standing, turned around, lie on the desk.

## Example

![](./assets/person-detection-action-recognition-0006.png)

## Specification

| Metric                            | Value                                     |
|-----------------------------------|-------------------------------------------|
| Detector AP (internal test set 2) | 90.70%                                    |
| Accuracy (internal test set 2)    | 80.74%                                    |
| Pose coverage                     | sitting, writing, raising_hand, standing, |
|                                   | turned around, lie on the desk            |
| Support of occluded pedestrians   | YES                                       |
| Occlusion coverage                | <50%                                      |
| Min pedestrian height             | 80 pixels (on 1080p)                      |
| GFlops                            | 8.225                                     |
| MParams                           | 2.001                                     |
| Source framework                  | TensorFlow\*                              |

Average Precision (AP) is defined as an area under the [precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve.

## Inputs

Image, name: `input`, shape: `1, 400, 680, 3` in the format `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order is `BGR`.

## Outputs

The net outputs four branches:

1. name: `ActionNet/out_detection_loc`, shape: `b, num_priors, 4` - Box coordinates in SSD format
2. name: `ActionNet/out_detection_conf`, shape: `b, num_priors, 2` - Detection confidences
3. name: `ActionNet/action_heads/out_head_1_anchor_1`, shape: `b, 50, 85, 6` - Action confidences
4. name: `ActionNet/action_heads/out_head_2_anchor_1`, shape: `b, 25, 43, 6` - Action confidences
5. name: `ActionNet/action_heads/out_head_2_anchor_2`, shape: `b, 25, 43, 6` - Action confidences
6. name: `ActionNet/action_heads/out_head_2_anchor_3`, shape: `b, 25, 43, 6` - Action confidences
7. name: `ActionNet/action_heads/out_head_2_anchor_4`, shape: `b, 25, 43, 6` - Action confidences

Where:

- `b` - batch size
- `num_priors` -  number of priors in SSD format (equal to 50x85x1+25x43x4=8550)

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Smart Classroom C++ Demo](../../../demos/smart_classroom_demo/cpp/README.md)
* [Smart Classroom C++ G-API Demo](../../../demos/smart_classroom_demo/cpp_gapi/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
