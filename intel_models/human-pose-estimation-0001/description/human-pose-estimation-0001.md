# human-pose-estimation-0001

## Use Case and High-Level Description

This is a multi-person 2D pose estimation network (based on the OpenPose approach) with tuned MobileNet v1 as a feature extractor. It finds a human pose: body skeleton, which consists of keypoints and connections between them, for every person inside image. The pose may contain up to 18 keypoints: ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees and ankles.

## Example

![](./human-pose-estimation-0001.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Average Precision (AP)          | 42.8%                                     |
| GFlops                          | 15.435                                    |
| MParams                         | 4.099                                     |
| Source framework                | Caffe*                                    |

Average Precision metric described in [COCO Keypoint Evaluation site](http://cocodataset.org/#keypoints-eval).

Tested on a COCO validation subset from the original paper: Cao et al. ["Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"](https://arxiv.org/pdf/1611.08050.pdf).

## Performance

## Inputs

1. name: "input" , shape: [1x3x256x456] - An input image in the format [BxCxHxW],
  where:
    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width.
  Expected color order is BGR.

## Outputs

1. The net outputs two blobs with shapes: [1, 38, 32, 57] and [1, 19, 32, 57]. The first blob contains keypoint pairwise relations (part affinity fields), the second one contains keypoint heatmaps.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
