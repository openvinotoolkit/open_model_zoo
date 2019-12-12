# text-spotting-0001-detector

## Use case and High-level description

This is text spotting model that means it simultaneously detects and
recognizes text. The model detects symbol sequences separated by space and does
 recognition without using any dictionary. The model is built on top of Mask-RCNN
 framework with additional attention-based text recognition head.

Symbols set is alphanumeric: 0123456789abcdefghijklmnopqrstuvwxyz

This model is Mask-RCNN-based text detector with ResNet50 backbone and additional text features output.

## Example

![](./text-spotting-0001.png)

## Specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| Word spotting hmean ICDAR2015, w/o dictionary | 59.04%    |
| GFlops (detection part)                       | `TBD`     |
| MParams (detection part)                      | `TBD`     |
| Source framework                              | PyTorch\* |

*Hmean Word spotting* is defined and measured according to
[Incidental Scene Text (ICDAR2015) challenge](https://rrc.cvc.uab.es/?ch=4&com=introduction).

## Performance

## Inputs

1.	name: `im_data` , shape: [1x3x768x1280] - An input image in the format
    [1xCxHxW]. The expected channel order is BGR.
1.	name: `im_info`, shape: [1x3] - Image information: processed image height,
    processed image width and processed image scale
    w.r.t. the original image resolution.

## Outputs

1.	name: `classes`, shape: [100, ] - Contiguous integer class ID for every
    detected object, '0' for background, i.e. no object.
1.	name: `scores`, shape: [100, ] - Detection confidence scores in range [0, 1]
    for every object.
1.	name: `boxes`, shape: [100, 4] - Bounding boxes around every detected objects
    in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
1.	name: `raw_masks`, shape: [100, 2, 28, 28] - Segmentation heatmaps for all
    classes for every output bounding box.
1.  name: `text_features`, shape [100, 64, 28, 28] - Text features that is fed to
text recognition head.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
