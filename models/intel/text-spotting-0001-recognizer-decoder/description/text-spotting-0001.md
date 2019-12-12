# text-spotting-0001-recognizer-decoder

## Use case and High-level description

This is text spotting model that means it simultaneously detects and
recognizes text. The model detects symbol sequences separated by space and does
 recognition without using any dictionary. The model is built on top of Mask-RCNN
 framework with additional attention-based text recognition head.

Symbols set is alphanumeric: 0123456789abcdefghijklmnopqrstuvwxyz

This model is 2d-attention-based GRU decoder of text recognition head.

## Example

![](../text-spotting-0001-detector/text-spotting-0001.png)

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

1.	name: `encoder_outputs` , shape: [1x256x64x64] - Encoded text recognition features.
1.	name: `prev_symbol` , shape: [1x1] - Index in alphabet of previously generated symbol.
1.	name: `prev_hidden`, shape: [1x1x256] - Previous hidden state of GRU.

## Outputs

1.	name: `output`, shape: [1x256x64x64] - Encoded text recognition features.
1.	name: `hidden`, shape: [1x1x256] - Current hidden state of GRU.


## Legal Information
[*] Other names and brands may be claimed as the property of others.
