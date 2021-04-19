# age-gender-recognition-retail-0013

## Use Case and High-Level Description

Fully convolutional network for simultaneous Age/Gender recognition. The network
is able to recognize age of people in [18, 75] years old range, it is not
applicable for children since their faces were not in the training set.

## Validation Dataset - Internal

~20,000 unique subjects representing diverse ages, genders, and ethnicities.

## Example

| Input Image                                   | Result        |
|-----------------------------------------------|---------------|
| ![](./assets/age-gender-recognition-retail-0001.jpg) | Female, 18.97 |
| ![](./assets/age-gender-recognition-retail-0002.png) | Male, 26.52   |
| ![](./assets/age-gender-recognition-retail-0003.png) | Male, 33.41   |

## Specification

| Metric                | Value                   |
|-----------------------|-------------------------|
| Rotation in-plane     | ±45˚                    |
| Rotation out-of-plane | Yaw: ±45˚ / Pitch: ±45˚ |
| Min object width      | 62 pixels               |
| GFlops                | 0.094                   |
| MParams               | 2.138                   |
| Source framework      | Caffe\*                 |

## Accuracy

| Metric          | Value      |
|-----------------|------------|
| Avg. age error  | 6.99 years |
| Gender accuracy |     95.80% |

## Inputs

Image, name: `input`, shape: `1, 3, 62, 62` in `1, C, H, W` format, where:

- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Outputs

1. Name: `age_conv3`, shape: `1, 1, 1, 1` - Estimated age divided by 100.
2. Name: `prob`, shape: `1, 2, 1, 1` - Softmax output across 2 type classes [0 - female, 1 - male].

## Legal Information
[*] Other names and brands may be claimed as the property of others.
