# smartlab-action-recognition-0001 (decoder)

## Use Case and High-Level Description

This is the decoder part of smartlab action recogntion model.

This model classifies action from noise_action, put_take and adjust_rider.

## Decoder model specification

The smartlab-action-recognition-decoder-0001 is a fully connected decoder part which accepts features from top and front views, computed by encoder and predicts score for action across following label list: `no_action`, `noise_action`,  `adjust_rider`

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

## Legal Information

[*] Other names and brands may be claimed as the property of others.
