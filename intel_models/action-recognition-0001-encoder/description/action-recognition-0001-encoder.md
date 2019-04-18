# Model name: action-recognition-0001-encoder

## Use Case and High-Level Description

This is an general-purpose action recognition model for Kinetics-400 dataset. The model uses Video Transformer approach with ResNet34 encoder.
Please refer to the [kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset specification to see list of action that are recognised by this model.

This model is only encoder part of the whole pipeline. It accepts video frame and produces embedding.
Use action-recognition-0001-decoder to produce prediction from embeddings of 16 frames.
Video frames should be sampled to cover ~1 second fragment (i.e. skip every second frame in 30 fps video).

## Example

![](./demo.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Source framework                | PyTorch*                                  |
| GFlops                          | 7.340                                     |
| MParams                         | 21.276                                    |

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
