# text-recognition-resnet-fc

## Use-case and high-level description

`text-recognition-resnet-fc` is a simple and preformant scene text recognition model based on ResNet with Fully Connected text recognition head. Source implementation on a PyTorch\* framework could be found [here](https://github.com/Media-Smart/vedastr). Model is able to recognize alphanumeric text.

## Specification

| Metric           | Value                  |
| ---------------- | ---------------------- |
| Type             | Scene Text Recognition |
| GFLOPs           | 40.3704                |
| MParams          | 177.9668               |
| Source framework | PyTorch\*              |

## Accuracy

Alphanumeric subset of common scene text recognition benchmarks are used. For your convenience you can see dataset size. Note, that we use here ICDAR15 alphanumeric subset without irregular (arbitrary oriented, perspective or curved) texts. See details [here](https://arxiv.org/abs/1709.02054), section 4.1. All reported results are achieved without using any lexicon.

| Dataset  | Accuracy | Dataset size |
| -------- | -------- | ------------ |
| ICDAR-03 | 92.96%   | 867          |
| ICDAR-13 | 90.44%   | 1015         |
| ICDAR-15 | 77.58%   | 1811         |
| SVT      | 88.56%   | 647          |
| IIIT5K   | 88.83%   | 3000         |

## Input

Image, name: `input`, shape: `1, 1, 32, 100` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Note that the source image should be tight aligned crop with detected text converted to grayscale. Mean values: [127.5, 127.5, 127.5], scale factor for each channel: 127.5.

## Outputs

Output tensor, name: `output`, shape: `1, 26, 37` in the format `B, W, L`, where:

- `W` - output sequence length
- `B` - batch size
- `L` - confidence distribution across alphanumeric symbols:
  `[s]0123456789abcdefghijklmnopqrstuvwxyz`, where [s] - special end of sequence character for decoder.

The network output decoding process is pretty easy: get the argmax on `L` dimension, transform indices to letters and slice the resulting phrase on the first entry of `end-of-sequence` symbol.

## Use text-detection demo

Model is supported by [text-detection c++ demo](../../../demos/text_detection_demo/cpp/main.cpp). In order to use this model in the demo, user should pass the following options:
```
  -tr_pt_first
  -dt "simple"
```

For more information, please, see documentation of the demo.

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://github.com/Media-Smart/vedastr/blob/0fd2a0bd7819ae4daa2a161501e9f1c2ac67e96a/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).

[*] Other names and brands may be claimed as the property of others.
