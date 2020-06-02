# text-detection-0003

## Use Case and High-Level Description

Text detector based on [PixelLink](https://arxiv.org/abs/1801.01315) architecture with [MobileNetV2-like](https://arxiv.org/abs/1801.04381) as a backbone for indoor/outdoor scenes.

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

Name: `input`, shape: [1x3x768x1280] - An input image in the format [BxCxHxW],
where:

   - B - batch size
   - C - number of channels
   - H - image height
   - W - image width

Expected color order: BGR.

## Outputs

1. name: "model/link\_logits\_/add", shape: [1x16x192x320] - logits related to linkage between pixels and their neighbors.

2. name: "model/segm\_logits/add", shape: [1x2x192x320] - logits related to text/no-text classification for each pixel.

Refer to [PixelLink](https://arxiv.org/abs/1801.01315) and demos for details.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
