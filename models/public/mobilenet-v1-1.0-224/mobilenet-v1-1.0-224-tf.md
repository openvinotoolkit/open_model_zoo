# mobilenet-v1-1.0-224

## Use Case and High-Level Description

`mobilenet-v1-1.0-224` is one of MobileNets - small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models are used. For details see [paper](https://arxiv.org/abs/1704.04861)

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 1.148                                     |
| MParams                         | 4.222                                     |
| Source framework                | Tensorflow                                |

## Performance

## Input

### Original model

1. Name: `input` , shape: [1x224x224x3] - An input image in the format [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order - BGR.

### Converted model

1. Name: `input` , shape: [1x3x224x224] - An input image in the format [BxCxHxW], where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

## Output

### Original model

1. Name: `MobilenetV1/Predictions/Reshape_1`, contains probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format.

### Converted model

1. Name: `MobilenetV1/Predictions/Softmax`, shape: [1,1001] in [BxC] format, where:

    - B - batch size
    - C - vector of probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format.

## Legal Information
[https://raw.githubusercontent.com/tensorflow/models/master/LICENSE]()