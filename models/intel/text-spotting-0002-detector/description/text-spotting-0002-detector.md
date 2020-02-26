# text-spotting-0002-detector

## Use Case and High-Level Description

This is a text spotting model that simultaneously detects and
recognizes text. The model detects symbol sequences separated by space and performs
recognition without a dictionary. The model is built on top of the Mask-RCNN
framework with additional attention-based text recognition head.

Symbols set is alphanumeric: `0123456789abcdefghijklmnopqrstuvwxyz`.

This model is a Mask-RCNN-based text detector with ResNet50 backbone and additional text features output.

## Example

![](./text-spotting-0002.png)

## Specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| Word spotting hmean ICDAR2015, without a dictionary | 61.01%    |
| GFlops                                        | 185.169   |
| MParams                                       | 26.497    |
| Source framework                              | PyTorch\* |

*Hmean Word spotting* is defined and measured according to the
[Incidental Scene Text (ICDAR2015) challenge](https://rrc.cvc.uab.es/?ch=4&com=introduction).

## Performance

## Inputs

1.	Name: `im_data` , shape: [1x3x768x1280]. An input image in the [1xCxHxW] format.
    The expected channel order is BGR.
2.	Name: `im_info`, shape: [1x3]. Image information: processed image height,
    processed image width and processed image scale with respect to the original image resolution.

## Outputs

1.	Name: `classes`, shape: [100]. Contiguous integer class ID for every
    detected object, `0` for background (no object detected).
1.	Name: `scores`, shape: [100]. Detection confidence scores in the [0, 1] range
    for every object.
1.	Name: `boxes`, shape: [100x4]. Bounding boxes around every detected object
    in the (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
1.	Name: `raw_masks`, shape: [100x2x28x28]. Segmentation heatmaps for all
    classes for every output bounding box.
1.  Name: `text_features`, shape [100x64x28x28]. Text features that are fed to a text recognition head.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
