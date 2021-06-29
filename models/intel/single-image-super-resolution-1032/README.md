# single-image-super-resolution-1032

## Use Case and High-Level Description

[An Attention-Based Approach for Single Image Super Resolution](https://arxiv.org/abs/1807.06779) but with reduced number of
channels and changes in network architecture. It enhances the resolution of the input image by a factor of 4.

## Example

Low resolution:

![](./assets/street_480x270.png)

Bicubic interpolation:

![](./assets/x4c_street_480x270.png)

Super resolution:

![](./assets/x4_street_480x270.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| PSNR                            | 29.29 dB                                  |
| GFlops                          | 11.654                                    |
| MParams                         | 0.030                                     |
| Source framework                | PyTorch\*                                 |

For reference, PSNR for bicubic upsampling on test dataset is 26.79 dB.

## Inputs

1. Image, name: `0`, shape: `1, 3, 270, 480` in the format `B, C, H, W`, where:

    - `B` - batch size
    - `C` - number of channels
    - `H` - image height
    - `W` - image width

2. Bicubic interpolation of the input image, name: `1`, shape: `1, 3, 1080, 1920` in the format `B, C, H, W`, where:

    - `B` - batch size
    - `C` - number of channels
    - `H` - image height
    - `W` - image width

  Expected color order is `BGR`.

## Outputs

The net output is a blob with shapes `1, 3, 1080, 1920` that contains image after super resolution.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
