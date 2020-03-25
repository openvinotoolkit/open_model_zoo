# human-pose-estimation-3d-0001

## Use Case and High-Level Description

Hand pose estimation model based on the [Hand Keypoint Detection in Single Images using Multiview Bootstrapping](https://arxiv.org/pdf/1704.07809.pdf) papers.

## Example

![](./human-pose-estimation-3d-0001.jpg)

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| AP(coco orig)                                                 | 45.15                   |
| GFlops                                                        | 103.238598928           |
| MParams                                                       | 36.832324               |
| Source framework                                              | Caffe\*                 |

## Performance

## Inputs

### Original model

Name: `image`, shape: `[1x3x368x368]`. An input image in the `[BxCxHxW]` format,
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - RGB. Scale values - [255,255,255]

### Converted model

Name: "image" , shape: [1x3x368x368] - An input image in the format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

## Outputs

The net outputs three blobs with the following shapes: `[1, 22, 46, 46]`. (For every keypoint own heatmap)

## Legal Information

The original model is distributed under the
[License](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE).

[*] ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY.
