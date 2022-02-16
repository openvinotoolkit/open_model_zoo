# text-detection-0003

## Use Case and High-Level Description

Text detector based on [PixelLink](https://arxiv.org/abs/1801.01315) architecture with [MobileNetV2-like](https://arxiv.org/abs/1801.04381) as a backbone for indoor/outdoor scenes.

## Example

![](./assets/text-detection-0003.png)

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| F-measure (Harmonic mean of precision and recall on ICDAR2015)| 82.12%                  |
| GFlops                                                        | 51.256                  |
| MParams                                                       | 6.747                   |
| Source framework                                              | TensorFlow\*            |

## Inputs

Image, name: `Placeholder`, shape: `1, 768, 1280, 3` in the format `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `BGR`.

## Outputs

1. name: `model/link_logits_/add`, shape: `1, 192, 320, 16` - logits related to linkage between pixels and their neighbors.

2. name: `model/segm_logits/add`, shape: `1, 192, 320, 2` - logits related to text/no-text classification for each pixel.

Refer to [PixelLink](https://arxiv.org/abs/1801.01315) and demos for details.

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Text Detection C++ Demo](../../../demos/text_detection_demo/cpp/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
