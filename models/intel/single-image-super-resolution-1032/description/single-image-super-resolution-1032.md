# single-image-super-resolution-1032

## Use Case and High-Level Description

[An Attention-Based Approach for Single Image Super Resolution](https://arxiv.org/abs/1807.06779) but with reduced number of
channels and changes in network achitecture. It enhances the resolution of the input image by a factor of 4.

## Example

Low resolution:

![](./street_480x270.png)

Bicubic interpolation:

![](./x4c_street_480x270.png)

Super resolution:

![](./x4_street_480x270.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| PSNR                            | 29.29 dB                                  |
| GFlops                          | 11.654                                    |
| MParams                         | 0.030                                     |
| Source framework                | PyTorch*                                  |

For reference, PSNR for bicubic upsampling on test dataset is 26.79 dB.

## Performance

## Inputs

1. name: "0" , shape: [1x3x270x480] - An input image in the format [BxCxHxW],
  where:
    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

2. name: "1" , shape: [1x3x1080x1920] - Bicubic interpolation of the input image in the format [BxCxHxW],
  where:
    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width


  Expected color order is BGR.

## Outputs

1. The net outputs one blobs with shapes [1, 3, 1080, 1920] that contains image after super
   resolution.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
