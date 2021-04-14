# text-recognition-0012

## Use Case and High-Level Description

This is a network for text recognition scenario. It consists of VGG16-like backbone and bidirectional LSTM encoder-decoder.
The network is able to recognize case-insensitive alphanumeric text (36 unique symbols).

## Example

![](./assets/openvino.jpg) -> openvino

## Specification

| Metric                                         | Value              |
|------------------------------------------------|--------------------|
| Accuracy on the alphanumeric subset of ICDAR13 | 0.8818             |
| Text location requirements                     | Tight aligned crop |
| GFlops                                         | 1.485              |
| MParams                                        | 5.568              |
| Source framework                               | TensorFlow\*       |

## Inputs

Image, name: `Placeholder`, shape: `1, 1, 32, 120` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Note that the source image should be tight aligned crop with detected text converted to grayscale.

## Outputs

The net output is a blob with the shape `30, 1, 37` in the format `W, B, L`, where:

- `W` - output sequence length
- `B` - batch size
- `L` - confidence distribution across alphanumeric symbols: `0123456789abcdefghijklmnopqrstuvwxyz#`, where # - special blank character for CTC decoding algorithm.

The network output can be decoded by CTC Greedy Decoder or CTC Beam Search decoder.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
