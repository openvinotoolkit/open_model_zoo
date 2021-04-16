# person-attributes-recognition-crossroad-0238

## Use Case and High-Level Description

This model presents a person attributes classification algorithm analysis scenario.
The model consists of the Inception V3 backbone and a head. For an input image with a pedestrian the model returns 7 values
that are probabilities of the corresponding 7 attributes.

## Specification

| Metric                | Value                                                                                                 |
|-----------------------|-------------------------------------------------------------------------------------------------------|
| Pedestrian pose       | Standing person                                                                                       |
| Occlusion coverage    | <20%                                                                                                  |
| Min object width      | 80 pixels                                                                                             |
| Supported attributes  | `is_male`, `has_bag`, `has_hat`, `has_longsleeves`, `has_longpants`, `has_longhair`, `has_coat_jacket`|
| GFlops                | 1.034                                                                                                 |
| MParams               | 21.797                                                                                                |
| Source framework      | PyTorch\*                                                                                             |

## Accuracy

| Attribute         |  F1   |
|-------------------|-------|
| `is_male`         | 0.80  |
| `has_bag`         | 0.48  |
| `has_hat`         | 0.42  |
| `has_longsleeves` | 0.17  |
| `has_longpants`   | 0.75  |
| `has_longhair`    | 0.77  |
| `has_coat_jacket` |  NA   |

## Inputs

Image, name: `input`, shape: `1, 3, 160, 80` in the format `1, C, H, W`, where:

- `C` - number of channels
- `H` - image height
- `W` - image width

The expected color order is `BGR`.

## Outputs

The net output is a blob named `attributes` with shape `1, 7` across seven attributes:
[`is_male`, `has_bag`, `has_hat`, `has_longsleeves`, `has_longpants`, `has_longhair`,
 `has_coat_jacket`].
Value > 0.5 means that the corresponding attribute is present.

## Legal Information
[\*] Other names and brands may be claimed as the property of others.
