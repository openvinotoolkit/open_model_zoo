# human-pose-estimation-0007

## Use Case and High-Level Description

This is a multi-person 2D pose estimation network based on the EfficientHRNet approach (that follows the Associative Embedding framework).
For every person in an image, the network detects a human pose: a body skeleton consisting of keypoints and connections between them.
The pose may contain up to 17 keypoints: ears, eyes, nose, shoulders, elbows, wrists, hips, knees, and ankles.

## Example

![](./assets/human-pose-estimation-0007.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Average Precision (AP)          | 54.3%                                     |
| GFlops                          | 14.3253                                   |
| MParams                         | 8.1506                                    |
| Source framework                | PyTorch\*                                 |

Average Precision metric described in [COCO Keypoint Evaluation site](https://cocodataset.org/#keypoints-eval).

## Inputs

Image, name: `image`, shape: `1, 3, 448, 448` in the `B, C, H, W` format, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W `- image width

Expected color order is `BGR`.

## Outputs

The net outputs are two blobs:

1. `heatmaps` of shape `1, 17, 224, 224` containing location heatmaps for keypoints of all types. Locations that are filtered out by non-maximum suppression algorithm have negated values assigned to them.
2. `embeddings` of shape `1, 17, 224, 224, 1` containing associative embedding values, which are used for grouping individual keypoints into poses.

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Human Pose Estimation C++ Demo](../../../demos/human_pose_estimation_demo/cpp/README.md)
* [Human Pose Estimation Python\* Demo](../../../demos/human_pose_estimation_demo/python/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
