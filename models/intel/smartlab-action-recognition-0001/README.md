# smartlab-action-recognition-0001 (composite)

## Use Case and High-Level Description

This is an smartlab action recogntion model that recognizes smartlab actions.
The model uses smartlab dataset to predict actions.
The model is classfier for 3 class actions.

## Example of the input data
![](./assets/frame0001.jpg)

## Example of the output
Name: `decoder_hidden`, shape: `1, 3`. The foramt is:

[`has_action_conf_score`, `action_1_logits`, `action_2_logits`].

Detailed explaination is available in [smartlab-action-recognition-0001-decoder](./smartlab-action-recognition-0001-decoder/README.md)


## Legal Information

[*] Other names and brands may be claimed as the property of others.