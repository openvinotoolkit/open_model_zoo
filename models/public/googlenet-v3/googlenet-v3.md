# googlenet-v3

## Use Case and High-Level Description

The `googlenet-v3` model is the first of the Inception family of models designed to perform image classification. For details about this family of models, check out the [paper](https://arxiv.org/pdf/1602.07261.pdf).

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 11.469        |
| MParams           | 23.819        |
| Source framework  | Tensorflow\*  |

## Accuracy

## Performance

## Input

### Original Model

Image, name: `input`, shape: [1x299x299x3], format: [BxHxWxC],
   where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: RGB.
   Mean values: [127.5, 127.5, 127.5], scale factor for each channel: 127.5

### Converted Model

Image, name: `input`, shape: [1x3x299x299], format: [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: BGR.

## Output

Object classifier according to ImageNet classes, name: `InceptionV3/Predictions/Softmax`, shape: [1,1001] in [BxC] format, where:

    - B - batch size
    - C - vector of probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format.

## Legal Information

[https://raw.githubusercontent.com/tensorflow/models/master/LICENSE]()
