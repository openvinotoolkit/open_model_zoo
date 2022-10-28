# smartlab-action-recognition-0001 (composite)

## Use Case and High-Level Description

There are 3 models for smartlab action recogntion including two encoder models and one decoder model. 

These models are fine-tuned with smartlab dataset to predict actions and can classfy 3 types of action including "noise_action", "put_take" and "adjust_rider".

## Example of the input data
![](./assets/frame0001.jpg)

## Example of the output
Name: `decoder_hidden`, shape: `1, 3`. The foramt is:

[`has_action_conf_score`, `action_1_logits`, `action_2_logits`].

Detailed explaination is available in [smartlab-action-recognition-0001-decoder](./smartlab-action-recognition-0001-decoder/README.md)


## Legal Information

[*] Other names and brands may be claimed as the property of others.