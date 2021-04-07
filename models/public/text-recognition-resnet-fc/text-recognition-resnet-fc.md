# Resnet-FC text recognition model

## Use-case and high-level description

This is a scene text recognition model. Source implementation on a PyTorch\* framework could be found [here](https://github.com/Media-Smart/vedastr). Model is able to recognize alphanumeric text.

## Specification

| Metric           | Value                  |
| ---------------- | ---------------------- |
| Type             | Scene Text Recognition |
| GFLOPs           | 40.3704                |
| MParams          | 177.9668               |
| Source framework | PyTorch\*              |

## Accuracy

Alphanumeric subset of common scene text recognition benchmarks are used. For your convenience you can see dataset size. Note, that we use here ICDAR15 alphanumeric subset without irregular (arbitrary oriented, perspective or curved) texts. See details [here](https://arxiv.org/pdf/1709.02054.pdf), section 4.1. All reported results are achieved without using any lexicon.

| Dataset  | Accuracy | Dataset size |
| -------- | -------- | ------------ |
| ICDAR-03 | 92.85%   | 867          |
| ICDAR-13 | 90.94%   | 1015         |
| ICDAR-15 | 77.80%   | 1811         |
| SVT      | 88.41%   | 647          |
| IIIT5K   | 87.77%   | 3000         |

## Input

Input tensor is `input.0`.
Shape: `1, 1, 32, 100` - An input image in the format `B, C, H, W`,
where:
  - B - batch size
  - C - number of channels
  - H - image height
  - W - image width

Note that the source image should be tight aligned crop with detected text converted to grayscale.

## Outputs
Output tensor is `output.0` with the shape `1, 26, 37` in the format `B, W, L`,
    where:
      - W - output sequence length
      - B - batch size
      - L - confidence distribution across alphanumeric symbols:
        "[s]0123456789abcdefghijklmnopqrstuvwxyz", where [s] - special end of sequence character for decoder.

The network output can be decoded by [simple decoder](../../../tools/accuracy_checker/accuracy_checker/adapters/text_recognition.py).

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://github.com/Media-Smart/vedastr/blob/0fd2a0bd7819ae4daa2a161501e9f1c2ac67e96a/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).

[*] Other names and brands may be claimed as the property of others.
