# smartlab-sequence-modelling-0001

## Use Case and High-Level Description
This is a feature extractor that is based on Mobilenet-v3 network without origianl classifier layer. Input is RGB image and output is feature vector.
For the original mobilenet-v3 model details see [PyTorch\* document](https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small) and [paper](https://arxiv.org/abs/1905.02244).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| GOPs                            | 0.11                                  |
| MParams                         | 2.537                                  |
| Source framework                | PyTorch\*                                 |



## Inputs

Image, name: `input`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

   - `B` - batch size
   - `C` - number of channels
   - `H` - image height
   - `W` - image width


## Outputs

Model has output name: `output`, shape: `1, 576, 1, 1`
`576` is the length of feature map.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
