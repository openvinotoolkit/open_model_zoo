# instance-segmentation-security-0050

## Use Case and High-Level Description

This model is an instance segmentation network for 80 classes of objects.
It is a Mask R-CNN with ResNet50 backbone, FPN and Bottom-Up Augmentation blocks
and light-weight RPN.

## Example

![](./instance-segmentation-security-0050.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| MS COCO val2017 box AP          | 31.27%                                    |
| MS COCO val2017 mask AP         | 27.83%                                    |
| Max objects to detect           | 100                                       |
| GFlops                          | 46.602                                    |
| MParams                         | 30.448                                    |
| Source framework                | PyTorch\*                                 |

Average Precision (AP) is defined and measured according to standard
[MS COCO evaluation procedure](http://cocodataset.org/#detection-eval).

## Performance

## Inputs

1.	name: `im_data` , shape: [1x3x480x480] - An input image in the format
    [1xCxHxW]. The expected channel order is BGR.
1.	name: `im_info`, shape: [1x3] - Image information: processed image height,
    processed image width and processed image scale
    w.r.t. the original image resolution.

## Outputs

1.	name: `classes`, shape: [100, ] - Contiguous integer class ID for every
    detected object, '0' for background, i.e. no object.
1.	name: `scores`: shape: [100, ] - Detection confidence scores in range [0, 1]
    for every object.
1.	name: `boxes`, shape: [100, 4] - Bounding boxes around every detected objects
    in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
1.	name: `raw_masks`, shape: [100, 81, 28, 28] - Segmentation heatmaps for all
    classes for every output bounding box.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
