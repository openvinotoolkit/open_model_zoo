# text-recognition-0016 (composite)

## Use Case and High-Level Description

This is a text-recognition composite model that recognizes scene text.
The model uses predefined set of alphanumeric symbols (case-insensitive) to predict words.
The model is built on the ResNeXt-101 backbone with [TPS](https://arxiv.org/abs/1603.03915) module and additional 2d attention-based text recognition head.

## Example of the input data

![](./assets/text-recognition-0016.jpg)

## Example of the output

`openvino`

## Composite model specification

| Metric                                         | Value              |
| ---------------------------------------------- | ------------------ |
| Text location requirements                     | Tight aligned crop |
| Source framework                               | PyTorch\*          |
| Accuracy on the alphanumeric subset of ICDAR13 | 0.9685             |
| Accuracy on the alphanumeric subset of ICDAR03 | 0.9712             |
| Accuracy on the alphanumeric subset of ICDAR15 | 0.8675             |
| Accuracy on the alphanumeric subset of SVT     | 0.9474             |
| Accuracy on the alphanumeric subset of IIIT5K  | 0.9347             |

## Encoder model specification

The text-recognition-0016-encoder model is a ResNeXt-101 like backbone with TPS network and convolutional encoder part of the text recognition.

| Metric  | Value |
| ------- | ----- |
| GFlops  | 9.27  |
| MParams | 88.1  |

### Inputs

Image, name: `imgs`, shape: `1, 1, 64, 256` in the `B, C, H, W` format, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width


### Outputs

1.	Name: `decoder_hidden`, shape: `1, 1, 1024`. Initial context state of the GRU cell.
2.	Name: `features`, shape: `1, 36, 1024`. Features from encoder part of text recognition head.

## Decoder model specification

The text-recognition-0016-decoder model is a GRU based decoder with 2d attention module.

| Metric  | Value |
| ------- | ----- |
| GFlops  | 0.08  |
| MParams | 4.28  |

### Inputs

1.	Name: `decoder_input`, shape: `1`. Previous predicted letter.
2.	Name: `features`, shape: `1, 36, 1024`. Encoded features.
3.	Name: `hidden`, shape: `1, 1, 1024`. Current state of the decoder.

### Outputs

1.	Name: `decoder_hidden`, shape: `1, 1, 1024`. Current context state of the GRU cell.
2.	Name: `decoder_output`, shape: `1, 40`. Classification confidence scores in the [0, 1] range
    for every letter.

Particularly, decoder output in every step is the probability distribution of the symbol on this timestamp. The model supports 40 symbols: 10 digits, 26 English alphabet letters and 4 special symbols(start of sequence symbol, end of sequence symbol, pad symbol and unknown symbol).
> Note: that start and end symbols are not passed in the supported symbol set in the demo, as well as pad symbol. See parameter `-m_tr_ss` in the demo section for details.
## Use text-detection demo

Model is supported by [text-detection c++ demo](../../../demos/text_detection_demo/cpp/README.md). In order to use this model in the demo, user should pass the following options:
```
-tr_pt_first
-m_tr_ss "?0123456789abcdefghijklmnopqrstuvwxyz"
-tr_o_blb_nm "decoder_output"
-tr_composite
-dt simple -lower
```

For more information, please, see documentation of the demo.

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Text Detection C++ Demo](../../../demos/text_detection_demo/cpp/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
