# driver-action-recognition-adas-0002 (composite)

## Use Case and High-Level Description

This is an action recognition composite model for the driver monitoring use case, consisting of encoder and decoder parts. The encoder model uses Video Transformer approach with MobileNetv2 encoder. It is able to recognize the following actions: drinking, doing hair or making up, operating the radio, reaching behind, safe driving, talking on the phone, texting.

## Example

![](./action-recognition-kelly.png)

## Composite model specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Source framework                | PyTorch*                                  |


## Encoder model specification

The driver-action-recognition-adas-0002-encoder model accepts video frame and produces embedding.
Video frames should be sampled to cover ~1 second fragment (i.e. skip every second frame in 30 fps video).

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| GFlops                          | 0.676                                     |
| MParams                         | 2.863                                     |


### Performance

### Inputs

1. name: "0" , shape: [1x3x224x224] - An input image in the format [BxCxHxW],
   where:
    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order is BGR.

### Outputs

The model outputs a tensor with the shape [1x512x1x1], representing embedding of precessed frame.


## Decoder model specification

The driver-action-recognition-adas-0002-decoder model accepts stack of frame embeddings, computed by driver-action-recognition-adas-0002-encoder, and produces prediction on input video.

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| GFlops                          | 0.147                                     |
| MParams                         | 4.205                                     |


### Performance

### Inputs

1. name: "0" , shape: [1x16x512] - An embedding image in the format [BxTxC],
   where:
    - B - batch size.
    - T - Duration of input clip.
    - C - dimension of embedding.

### Outputs

The model outputs a tensor with the shape [bx9], each row is a logits vector of performed actions.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
