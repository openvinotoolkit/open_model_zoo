# text-recognition-0015 (composite)

## Use Case and High-Level Description

This is an text-recognition composite model that recognizes scene text.
The model uses predefined set of alphanumeric symbols (case-sensitive) to predict words.
The model is built on the ResNeXt-101 backbone with additional 2d attention-based text recognition head.

## Example of the input data

![](./assets/openvino.jpg)

## Example of the output

`openvino`

## Composite model specification

| Metric                                         | Value              |
| ---------------------------------------------- | ------------------ |
| Accuracy on the alphanumeric subset of ICDAR13 | 0.8995             |
| Accuracy on the alphanumeric subset of ICDAR03 | 0.9389             |
| Accuracy on the alphanumeric subset of ICDAR15 | 0.7355             |
| Accuracy on the alphanumeric subset of SVT     | 0.8764             |
| Accuracy on the alphanumeric subset of IIIT5K  | 0.8413             |
| Text location requirements                     | Tight aligned crop |
| Source framework                               | PyTorch\*          |

The above accuracies are calculated for case-insensitive mode (i.e. GT text and predicted text are all casted to lowercase).

## Encoder model specification

The text-recognition-0015-encoder model is a ResNeXt-101 like backbone with convolutional encoder part of the text recognition.

| Metric  | Value |
| ------- | ----- |
| GFlops  | 12.4  |
| MParams | 398   |

### Inputs

Image, name: `imgs`, shape: `1, 1, 64, 256` in the `1, C, H, W` format, where:

- `C` - number of channels
- `H` - image height
- `W` - image width


### Outputs

1.	Name: `decoder_hidden`, shape: `1, 1, 1024`. Initial context state of the GRU cell.
2.	Name: `features`, shape: `1, 16, 1024`. Features from encoder part of text recognition head.

## Decoder model specification

The text-recognition-15-decoder model is a GRU based decoder with 2d attention module.

| Metric  | Value |
| ------- | ----- |
| GFlops  | 0.03  |
| MParams | 4.33  |

### Inputs

1.	Name: `decoder_input`, shape: `1`. Previous predicted letter.
2.	Name: `features`, shape: `1, 16, 1024`. Encoded features.
3.	Name: `hidden`, shape: `1, 1, 1024`. Current state of the decoder.

### Outputs

1.	Name: `decoder_hidden`, shape: `1, 1, 1024`. Current context state of the LSTM cell.
2.	Name: `decoder_output`, shape: `1, 66`. Classification confidence scores in the [0, 1] range
    for every letter.
## Use text-detection demo

Model is supported by [text-detection c++ demo](../../../demos/text_detection_demo/cpp/README.md). In order to use this model in the demo, user should pass the following options:
```
-tr_pt_first
-m_tr_ss "?0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
-tr_o_blb_nm "logits"
-tr_composite
-dt simple -lower
```

For more information, please, see documentation of the demo.
## Legal Information
[*] Other names and brands may be claimed as the property of others.
