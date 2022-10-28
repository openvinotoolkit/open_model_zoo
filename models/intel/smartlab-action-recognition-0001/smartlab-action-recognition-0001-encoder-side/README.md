# smartlab-action-recognition-0001 (encoder-side)

## Use Case and High-Level Description

This is the encoder-side part of smartlab action recogntion model.

This model encodes the side-view of image frame into feature vector.

## Encoder model specification

The smartlab-action-recognition-encoder-0001 is a Mobilenet-V2 like backbone with convolutional encoder part of the action recognition.

| Metric  | Value |
| ------- | ----- |
| GFlops  | 0.611 |
| MParams | 3.387 |

### Inputs

Image, name: `input_image`, shape: `1, 3, 224, 224` in the `B, C, H, W` format, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width
Expected color order is `BGR`

### Outputs

1. Name: `output_feature`, shape: `1, 1280`. Features from encoder part of action recogntion head.

## Legal Information

[*] Other names and brands may be claimed as the property of others.