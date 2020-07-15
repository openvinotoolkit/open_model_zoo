# emotions-recognition-retail-0003

## Use Case and High-Level Description

Fully convolutional network for recognition of five emotions ('neutral', 'happy', 'sad', 'surprise', 'anger').

## Validation Dataset

For the metrics evaluation, the validation part of
the [AffectNet](http://mohammadmahoor.com/affectnet/) dataset is used. A subset with
only the images containing five aforementioned emotions is chosen. The total amount of the images used in validation is 2,500.

## Example

| Input Image                                 | Result        |
|---------------------------------------------|---------------|
| ![](./emotions-recognition-retail-0003.jpg) | Happiness     |

## Specification

| Metric                | Value                   |
|-----------------------|-------------------------|
| Input face orientation| Frontal                 |
| Rotation in-plane     | ±15˚                    |
| Rotation out-of-plane | Yaw: ±15˚ / Pitch: ±15˚ |
| Min object width      | 64 pixels               |
| GFlops                | 0.126                   |
| MParams               | 2.483                   |
| Source framework      | Caffe                   |

## Accuracy

| Metric          | Value      |
|-----------------|------------|
| Accuracy        |     70.20% |

## Performance

## Inputs

Name: `input`, shape: [1x3x64x64] - An input image in [1xCxHxW] format. Expected color order is BGR.

## Outputs

1. name: "prob", shape: [1, 5, 1, 1] - Softmax output across five emotions
   ('neutral', 'happy', 'sad', 'surprise', 'anger').

## Legal Information
[*] Other names and brands may be claimed as the property of others.
