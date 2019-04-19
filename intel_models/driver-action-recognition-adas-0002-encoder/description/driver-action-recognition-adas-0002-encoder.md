# Model name: driver-action-recognition-adas-0002-encoder

## Use Case and High-Level Description

This is an action recognition model for the driver monitoring use case. The model uses Video Transformer approach with MobileNetv2 encoder. It is able to recognize the following actions: drinking, doing hair or making up, operating the radio, reaching behind, safe driving, talking on the phone, texting.

This model is only encoder part of the whole pipeline. It accepts video frame and produces embedding. Use driver-action-recognition-adas-0002-decoder to produce prediction from embeddings of 16 frames. Video frames should be sampled to cover ~1 second fragment (i.e. skip every second frame in 30 fps video).

## Example

![](./action-recognition-kelly.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Source framework                | PyTorch*                                  |
| GFlops                          | 0.676                                     |
| MParams                         | 2.863                                     |

## Performance

## Inputs

1. name: "0" , shape: [1x3x224x224] - An input image in the format [BxCxHxW],
   where:
    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order is BGR.

## Outputs

The model outputs a tensor with the shape [1x512x1x1], representing embedding of precessed frame.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
