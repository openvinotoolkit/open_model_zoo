# text-spotting-0004 (composite)

## Use Case and High-Level Description

This is a text spotting composite model that simultaneously detects and
recognizes text. The model detects symbol sequences separated by space and performs
recognition without a dictionary. The model is built on top of the Mask-RCNN
framework with additional attention-based text recognition head.

Alphabet is alphanumeric: `abcdefghijklmnopqrstuvwxyz0123456789`.

## Example

![](./text-spotting-0004.png)

## Composite model specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| Word spotting hmean ICDAR2015, without a dictionary | 70.34% |
| Source framework                              | PyTorch\* |

*Hmean Word spotting* is defined and measured according to the
[Incidental Scene Text (ICDAR2015) challenge](https://rrc.cvc.uab.es/?ch=4&com=introduction).

## Detector model specification

The text-spotting-0004-detector model is a Mask-RCNN-based text detector with ResNet50 backbone and additional text features output.

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 184.495   |
| MParams                                       | 27.010    |

### Inputs

1.	Name: `im_data` , shape: `1, 3, 768, 1280`. An input image in the `1, C, H, W` format.
    The expected channel order is BGR.
2.	Name: `im_info`, shape: `1, 3`. Image information: processed image height,
    processed image width, and processed image scale with respect to the original image resolution.

### Outputs

1.	Name: `labels`, shape: `100`. Contiguous integer class ID for every
    detected object, `0` is for text class.
2.	Name: `boxes`, shape: `100, 5`. Bounding boxes around every detected object
    in the (top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence) format.
3.	Name: `masks`, shape: `100, 28, 28`. Text segmentation masks for every output bounding box.
4.  Name: `text_features.0`, shape `100, 64, 28, 28`. Text features that are fed to a text recognition head.

## Encoder model specification

The text-spotting-0004-recognizer-encoder model is a fully-convolutional encoder of text recognition head.

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 2.082     |
| MParams                                       | 1.328     |

### Inputs

Name: `input` , shape: `1, 64, 28, 28`. Text recognition features obtained from detection part.

### Outputs

Name: `output`, shape: `1, 256, 28, 28`. Encoded text recognition features.

## Decoder model specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| GFlops                                        | 0.106     |
| MParams                                       | 0.283     |

### Inputs

1.	Name: `encoder_outputs` , shape: `1, (28*28), 256`. Encoded text recognition features.
1.	Name: `prev_symbol` , shape: `1, 1`. Index in alphabet of previously generated symbol.
1.	Name: `prev_hidden`, shape: `1, 1, 256`. Previous hidden state of GRU.

### Outputs

1.	Name: `output`, shape: `1, 38`. Encoded text recognition features. Indices starting from 2 correspond to symbols from the
alphabet. The 0 and 1 are special Start of Sequence and End of Sequence symbols correspondingly.
1.	Name: `hidden`, shape: `1, 1, 256`. Current hidden state of GRU.


## Legal Information
[*] Other names and brands may be claimed as the property of others.
