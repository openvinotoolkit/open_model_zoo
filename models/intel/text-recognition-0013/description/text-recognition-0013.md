# text-recognition-0013

## Use Case and High-Level Description

This is a network for text recognition scenario. It consists of ResNext50-like backbone (stage-1-2) and bidirectional LSTM encoder-decoder.
The network is able to recognize case-insensitive alphanumeric text (36 unique symbols).

## Example

![](./openvino.jpg) -> openvino

## Specification

| Metric                                         | Value              |
| ---------------------------------------------- | ------------------ |
| Accuracy on the alphanumeric subset of ICDAR13 | 0.8828             |
| Text location requirements                     | Tight aligned crop |
| GFlops                                         | 0.2726             |
| MParams                                        | 1.4187             |
| Source framework                               | PyTorch            |

## Inputs
Input tensor is `imgs`.
Shape: `1, 1, 32, 120` - An input image in the format `B, C, H, W`,
where:
  - B - batch size
  - C - number of channels
  - H - image height
  - W - image width

Note that the source image should be tight aligned crop with detected text converted to grayscale.

## Outputs
The net outputs 2 blobs
*  `logits` with the shape `30, 1, 37` in the format `W, B, L`,
    where:
      - W - output sequence length
      - B - batch size
      - L - confidence distribution across alphanumeric symbols: "#0123456789abcdefghijklmnopqrstuvwxyz", where # - special blank character for CTC decoding algorithm.

The network output can be decoded by CTC Greedy Decoder or CTC Beam Search decoder.

## Use text-detection demo

Model is supported by [text-detection c++ demo](../../../../demos/text_detection_demo/cpp/main.cpp). In order to use this model in the demo, user should pass the following options:
```
  -tr_pt_first
  tr_o_blb_nm "logits"
```

For more information, please, see documentation of the demo.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
