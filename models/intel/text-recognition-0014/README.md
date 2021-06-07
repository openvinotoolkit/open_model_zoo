# text-recognition-0014

## Use Case and High-Level Description

This is a network for text recognition scenario. It consists of ResNext101-like backbone (stage-1-2) and bidirectional LSTM encoder-decoder.
The network is able to recognize case-insensitive alphanumeric text (36 unique symbols).

## Example

![](./assets/openvino.jpg) -> openvino

## Specification

| Metric                                         | Value              |
| ---------------------------------------------- | ------------------ |
| Accuracy on the alphanumeric subset of ICDAR13 | 0.8887             |
| Accuracy on the alphanumeric subset of ICDAR03 | 0.9077             |
| Accuracy on the alphanumeric subset of ICDAR15 | 0.6908             |
| Accuracy on the alphanumeric subset of SVT     | 0.83               |
| Accuracy on the alphanumeric subset of IIIT5K  | 0.8157             |
| Text location requirements                     | Tight aligned crop |
| GFlops                                         | 0.2726             |
| MParams                                        | 1.4187             |
| Source framework                               | PyTorch\*          |

## Inputs

Image, name: `imgs`, shape: `1, 1, 32, 128` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Note that the source image should be tight aligned crop with detected text converted to grayscale.

## Outputs

The net output is a blob with name `logits` and the shape `16, 1, 37` in the format `W, B, L`, where:

- `W` - output sequence length
- `B` - batch size
- `L` - confidence distribution across alphanumeric symbols: `#0123456789abcdefghijklmnopqrstuvwxyz`, where # - special blank character for CTC decoding algorithm.

The network output can be decoded by CTC Greedy Decoder or CTC Beam Search decoder.

## Use text-detection demo

Model is supported by [text-detection c++ demo](../../../demos/text_detection_demo/cpp/README.md). In order to use this model in the demo, user should pass the following options:
```
  -tr_pt_first
  -tr_o_blb_nm "logits"
```

For more information, please, see documentation of the demo.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
