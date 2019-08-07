# text-detection-0003

## Use Case and High-Level Description

Text detector based on [PixelLink](https://arxiv.org/pdf/1801.01315.pdf) architecture with [MobileNetV2-like](https://arxiv.org/pdf/1801.04381.pdf) as a backbone for indoor/outdoor scenes.

## Example

![](./text-detection-0003.png)

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| F-measure (Harmonic mean of precision and recall on ICDAR2015)| 82.12%                  |
| GFlops                                                        | 51.256                  |
| MParams                                                       | 6.747                   |
| Source framework                                              | TensorFlow              |

## Performance

## Inputs

1. name: "input" , shape: [1x3x768x1280] - An input image in the format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

## Outputs

1. The net outputs two blobs. Refer to [PixelLink](https://arxiv.org/pdf/1801.01315.pdf) and demos for details.
    - [1x2x192x320] - logits related to text/no-text classification for each pixel.
    - [1x16x192x320] - logits related to linkage between pixels and their neighbors.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
