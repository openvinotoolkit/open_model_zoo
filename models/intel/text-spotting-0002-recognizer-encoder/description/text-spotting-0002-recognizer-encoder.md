# text-spotting-0002-recognizer-encoder

## Use Case and High-Level Description

This is a text spotting model that simultaneously detects and
recognizes text. The model detects symbol sequences separated by space and performs
recognition without a dictionary. The model is built on top of the Mask-RCNN
framework with additional attention-based text recognition head.

Symbols set is alphanumeric: `0123456789abcdefghijklmnopqrstuvwxyz`.

This model is a fully-convolutional encoder of text recognition head.

## Example

![](./text-spotting-0002.png)

## Specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| Word spotting hmean ICDAR2015, without a dictionary | 59.04%    |
| GFlops                                        | 2.082     |
| MParams                                       | 1.328     |
| Source framework                              | PyTorch\* |

*Hmean Word spotting* is defined and measured according to the
[Incidental Scene Text (ICDAR2015) challenge](https://rrc.cvc.uab.es/?ch=4&com=introduction).

## Performance

## Inputs

Name: `input` , shape: [1x64x28x28]. Text recognition features obtained from detection part.

## Outputs

Name: `output`, shape: [1x256x28x28]. Encoded text recognition features.


## Legal Information
[*] Other names and brands may be claimed as the property of others.
