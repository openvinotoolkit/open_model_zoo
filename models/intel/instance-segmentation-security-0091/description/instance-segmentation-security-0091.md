# instance-segmentation-security-0091

## Use Case and High-Level Description

This model is an instance segmentation network for 80 classes of objects.
It is a Cascade mask R-CNN with ResNet101 backbone and deformable convolutions,
FPN, RPN, detection and segmentation heads.

## Example

![](./instance-segmentation-security-0091.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| MS COCO val2017 box AP          | 45.8%                                     |
| MS COCO val2017 mask AP         | 39.7%                                     |
| Max objects to detect           | 1000                                      |
| GFlops                          | 828.6324                                  |
| MParams                         | 101.236                                   |
| Source framework                | PyTorch\*                                 |

Average Precision (AP) is defined and measured according to standard
[MS COCO evaluation procedure](http://cocodataset.org/#detection-eval).

## Performance

## Inputs

1.	name: `image` , shape: [1x3x800x1344] - An input image in the format
    [1xCxHxW]. The expected channel order is BGR.

## Outputs

1.	name: `labels`, shape: [1000, ] - Contiguous integer class ID for every
    detected object.
2.	name: `boxes`, shape: [1000, 5] - Bounding boxes around every detected objects
    in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format and its
    confidence score in range [0, 1].
3.	name: `masks`, shape: [1000, 28, 28] - Segmentation heatmaps for every output
    bounding box.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
