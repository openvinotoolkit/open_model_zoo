# smartlab-action-recognition-0001 (composite)

## Use Case and High-Level Description

This is an smartlab action recogntion model that recognizes smartlab actions.
The model uses smartlab dataset to predict actions.
The model is classfier for 3 class actions.

## Example of the input data
<!-- ![](./assets/frame0001.jpg) -->

## Example of the output

`openvino`

## Composite model specification

| Metric                                         | Value              |
| ---------------------------------------------- | ------------------ |
| Accuracy on the DSI1867                        | TODO               |
| Source framework                               | PyTorch\*          |


## Encoder model specification

The smartlab-action-recognition-encoder-0001 is a Mobilenet-V2 like backbone with convolutional encoder part of the action recognition.

| Metric  | Value |
| ------- | ----- |
| GFlops  | 0.611 |
| MParams | 3.387 |

### Inputs

Image, name: `input_image`, shape: `1, 1, 224, 224` in the `1, C, H, W` format, where:

- `C` - number of channels
- `H` - image height
- `W` - image width


### Outputs

1.	Name: `output_feature`, shape: `1, 1280`. Features from encoder part of action recogntion head.

## Decoder model specification

The smartlab-action-recognition-decoder-0001 is a fully connected layer module.

| Metric  | Value |
| ------- | ----- |
| GFlops  | 0.008 |
| MParams | 4.099 |

### Inputs

1.	Name: `input_feature_1`, shape: `1, 1280`. Encoded features from topview.
2.	Name: `input_feature_1`, shape: `1, 1280`. Encoded features from frontview.

### Outputs

1.	Name: `decoder_hidden`, shape: `1, 3`. The foramt [`has_action_conf_score`, `action_1_logits`, `action_2_logits`]

- `has_action_conf_score` - confidence for action frame
- `noise_action` - confidence for the noise_action class
- `adjust_rider` - confidence for the adjust_rider class

Classification confidence scores in the [0, 1] range
    for every smartlab actions.

## Use smartlab demo

Model is supported by [smartlab_demo python demo](../../../demos/smartlab_demo/python/README.md).


For more information, please, see documentation of the demo.
## Legal Information
[*] Other names and brands may be claimed as the property of others.
