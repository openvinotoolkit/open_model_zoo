# smartlab-action-recognition-0001 (composite)

## Use Case and High-Level Description

There are 3 models for smartlab action recogntion including two encoder models and one decoder model.

These models are fine-tuned with smartlab dataset to predict actions and can classfy 3 types of action including "noise_action", "put_take" and "adjust_rider".

## Example of the input data
![](./assets/frame0001.jpg)

## Example of the output
Output `put_take` action

## Composite model specification
| Metric                                         | Value              |
| ---------------------------------------------- | ------------------ |
| Accuracy on the DSI1867                        | TODO               |
| Source framework                               | PyTorch\*          |

## Encoder models specification

The smartlab-action-recognition-0001-encoder-* have Mobilenet-V2 like backbone with convolutional encoder part of the action recognition.

There are two models called: `smartlab-action-recognition-0001-encoder-side` and `smartlab-action-recognition-0001-encoder-top`, which have the same strcuture but different weights.

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

## Decoder model specification

The smartlab-action-recognition-0001-decoder is a fully connected decoder part which accepts features from top and front views, computed by encoder and predicts score for action across following label list: `no_action`, `noise_action`,  `adjust_rider`

| Metric  | Value |
| ------- | ----- |
| GFlops  | 0.008 |
| MParams | 4.099 |

### Inputs

1. Name: `input_feature_1`, shape: `1, 1280`. Encoded features from topview.
2. Name: `input_feature_2`, shape: `1, 1280`. Encoded features from frontview.

### Outputs

1. Name: `decoder_hidden`, shape: `1, 3`. The foramt [`has_action_conf_score`, `action_1_logits`, `action_2_logits`]
    * `has_action_conf_score` - confidence for action frame. If>0.5, there is specified action.
    * `action_1_logits` - confidence for the put_take action class
    * `action_2_logits` - confidence for the adjust_rider action class

Classification confidence scores in the [0, 1] range.
## Demo usage
The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

- [smartlab_demo/python](../../../demos/smartlab_demo/python/README.md)

## Legal Information

[*] Other names and brands may be claimed as the property of others.
