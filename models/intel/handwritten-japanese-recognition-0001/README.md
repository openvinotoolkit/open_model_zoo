# handwritten-japanese-recognition-0001

## Use Case and High-Level Description

This is a network for handwritten Japanese text recognition scenario. It consists of a VGG16-like backbone,
reshape layer and a fully connected layer.
The network is able to recognize Japanese text consisting of characters in the [Kondate](http://web.tuat.ac.jp/~nakagawa/database/en/kondate_about.html) and [Nakayosi](http://web.tuat.ac.jp/~nakagawa/database/en/about_nakayosi.html) datasets.

## Example

![](./assets/test.png) -> 菊池朋子

## Specification

| Metric                                                            | Value              |
|-------------------------------------------------------------------|--------------------|
| GFlops                                                            | 117.136            |
| MParams                                                           | 15.31              |
| Accuracy on Kondate test set and test set generated from Nakayosi | 98.16%             |
| Source framework                                                  | PyTorch\*          |

## Accuracy Values

This demo adopts [label error rate](https://dl.acm.org/doi/abs/10.1145/1143844.1143891) as the metric for accuracy.

## Inputs

Grayscale image, name - `actual_input`, shape - `1, 1, 96, 2000`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

> **NOTE:**  the source image should be resized to specific height (such as 96) while keeping aspect ratio, and the width after resizing should be no larger than 2000 and then the width should be right-bottom padded to 2000 with edge values.

## Outputs

Name - `output`, shape - `186, 1, 4442`, format is `W, B, L`, where:

- `W` - output sequence length
- `B` - batch size
- `L` - confidence distribution across the supported symbols in [Kondate](http://web.tuat.ac.jp/~nakagawa/database/en/kondate_about.html) and [Nakayosi](http://web.tuat.ac.jp/~nakagawa/database/en/about_nakayosi.html).

The network output can be decoded by CTC Greedy Decoder.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
