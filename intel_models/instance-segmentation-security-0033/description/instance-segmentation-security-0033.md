# instance-segmentation-security-0033

## Use case and High-level description

This model is an instance segmentation network for 80 classes of objects.
It is a Mask-RCNN-like model with ResNeXt152 backbone and
Feature Pyramid Networks block for feature maps refinement.

## Example

![](./instance-segmentation-security-0033.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| MS COCO val2017 box AP (max short side 480, max long side 640)   | 38.9%    |
| MS COCO val2017 mask AP (max short side 480, max long side 640)  | 34.7%    |
| MS COCO val2017 box AP (max height 480, max width 640)           | 38.6%    |
| MS COCO val2017 mask AP (max height 480, max width 640)          | 34.3%    |
| Max objects to detect           | 100                                       |
| GFlops                          | 354.274                                   |
| MParams                         | 143.444                                   |
| Source framework                | PyTorch\*                                 |

Average Precision (AP) is defined and measured according to standard
[MS COCO evaluation procedure](http://cocodataset.org/#detection-eval).

## Performance

## Inputs

1.	name: `im_data` , shape: [1x3x480x640] - An input image in the format
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
