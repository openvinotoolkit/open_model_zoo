# text-spotting-0002-recognizer-decoder

## Use Case and High-Level Description

This is a text spotting model that simultaneously detects and
recognizes text. The model detects symbol sequences separated by space and performs
recognition without a dictionary. The model is built on top of the Mask-RCNN
framework with additional attention-based text recognition head.

Symbols set is alphanumeric: `0123456789abcdefghijklmnopqrstuvwxyz`.

This model is 2D attention-based GRU decoder of text recognition head.


## Example

![](./text-spotting-0002.png)

## Specification

| Metric                                        | Value     |
|-----------------------------------------------|-----------|
| Word spotting hmean ICDAR2015, without a dictionary | 59.04%    |
| GFlops                                        | 0.002     |
| MParams                                       | 0.273     |
| Source framework                              | PyTorch\* |

*Hmean Word spotting* is defined and measured according to the
[Incidental Scene Text (ICDAR2015) challenge](https://rrc.cvc.uab.es/?ch=4&com=introduction).

## Performance

## Inputs

1.	Name: `encoder_outputs` , shape: [1x(28*28)x256]. Encoded text recognition features.
1.	Name: `prev_symbol` , shape: [1x1]. Index in alphabet of previously generated symbol.
1.	Name: `prev_hidden`, shape: [1x1x256]. Previous hidden state of GRU.

## Outputs

1.	Name: `output`, shape: [1x38]. Encoded text recognition features.
1.	Name: `hidden`, shape: [1x1x256]. Current hidden state of GRU.


## Legal Information

[*] Other names and brands may be claimed as the property of others.
