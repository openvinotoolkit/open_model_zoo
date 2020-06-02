# text-spotting-0002 (composite)

## Use Case and High-Level Description

This is a text spotting composite model that simultaneously detects and
recognizes text. The model detects symbol sequences separated by space and performs
recognition without a dictionary. The model is built on top of the Mask-RCNN
framework with additional attention-based text recognition head.

Symbols set is alphanumeric: `0123456789abcdefghijklmnopqrstuvwxyz`.

## Example

![](./text-spotting-0002.png)

## Composite model specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| Word spotting hmean ICDAR2015, without a dictionary | 61.01% |
| Source framework                              | PyTorch\* |

*Hmean Word spotting* is defined and measured according to the
[Incidental Scene Text (ICDAR2015) challenge](https://rrc.cvc.uab.es/?ch=4&com=introduction).

## Detector model specification

The text-spotting-0002-detector model is a Mask-RCNN-based text detector with ResNet50 backbone and additional text features output.

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 185.169   |
| MParams                                       | 26.497    |


### Performance

### Inputs

1.	Name: `im_data` , shape: [1x3x768x1280]. An input image in the [1xCxHxW] format.
    The expected channel order is BGR.
2.	Name: `im_info`, shape: [1x3]. Image information: processed image height,
    processed image width, and processed image scale with respect to the original image resolution.

### Outputs

1.	Name: `classes`, shape: [100]. Contiguous integer class ID for every
    detected object, `0` for background (no object detected).
2.	Name: `scores`, shape: [100]. Detection confidence scores in the [0, 1] range
    for every object.
3.	Name: `boxes`, shape: [100x4]. Bounding boxes around every detected object
    in the (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
4.	Name: `raw_masks`, shape: [100x2x28x28]. Segmentation heatmaps for all
    classes for every output bounding box.
5.  Name: `text_features`, shape [100x64x28x28]. Text features that are fed to a text recognition head.


## Encoder model specification

The text-spotting-0002-recognizer-encoder model is a fully-convolutional encoder of text recognition head.

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 2.082     |
| MParams                                       | 1.328     |


### Performance

### Inputs

Name: `input` , shape: [1x64x28x28]. Text recognition features obtained from detection part.

### Outputs

Name: `output`, shape: [1x256x28x28]. Encoded text recognition features.


## Decoder model specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 0.002     |
| MParams                                       | 0.273     |


### Performance

### Inputs

1.	Name: `encoder_outputs` , shape: [1x(28*28)x256]. Encoded text recognition features.
1.	Name: `prev_symbol` , shape: [1x1]. Index in alphabet of previously generated symbol.
1.	Name: `prev_hidden`, shape: [1x1x256]. Previous hidden state of GRU.

### Outputs

1.	Name: `output`, shape: [1x38]. Encoded text recognition features.
1.	Name: `hidden`, shape: [1x1x256]. Current hidden state of GRU.


## Legal Information
[*] Other names and brands may be claimed as the property of others.
