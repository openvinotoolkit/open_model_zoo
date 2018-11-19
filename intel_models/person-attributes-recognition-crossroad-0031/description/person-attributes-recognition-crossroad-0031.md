# person-attributes-recognition-crossroad-0031

## Use Case and High-Level Description

This model presents a person attributes classification algorithm for a traffic analysis scenario.

## Example
![](./person-attributes-recognition-crossroad-0031-1.png)

| Attribute       | Value   |
|-----------------|---------|
| is male         | true    |
| has hat         | false   |
| has longsleeves | true    |
| has longpants   | true    |
| has longhair    | false   |
| has coat_jacket | false   |

## Specification

| Metric                | Value                                                                          |
|-----------------------|--------------------------------------------------------------------------------|
| Pedestrian pose       | Standing person                                                                |
| Occlusion coverage    | <20%                                                                           |
| Min object width      | 80 pixels                                                                      |
| Supported attributes  | gender, has hat, has longsleeves, has longpants, has longhair, has coat_jacket |
| GFlops                | 0.219                                                                          |
| MParams               | 1.102                                                                          |
| Source framework      | Caffe*                                                                         |

## Accuracy

| Attribute         | Precision | Recall | F1   | Accuracy |
|-------------------|---------- |------- |----- |--------- |
| **Average**       | 0.90      | 0.86   | 0.88 | 0.90     |
| `is_male`         | 0.89      | 0.87   | 0.88 | 0.86     |
| `has_hat`         | 0.67      | 0.35   | 0.46 | 0.97     |
| `has_longsleeves` | 0.92      | 0.93   | 0.96 | 0.94     |
| `has_longpants`   | 0.97      | 0.96   | 0.96 | 0.94     |
| `has_longhair`    | 0.80      | 0.73   | 0.76 | 0.91     |
| `has_coat_jacket` | 0.73      | 0.52   | 0.61 | 0.84     |

## Performance
Link to [performance table](https://software.intel.com/en-us/openvino-toolkit/benchmarks)

## Inputs

1.	name: "input" , shape: [1x3x80x160] - An input image in following format
[1xCxHxW], where

	- C - number of channels
    	- H - image height
    	- W - image width.

	The expected color order is BGR.

## Outputs

1.	The net outputs a blob named sigmoid with shape: [1, 6, 1, 1] across six attributes:
    [`is_male`, `has_hat`, `has_longsleeves`, `has_longpants`, `has_longhair`,
     `has_coat_jacket`]. Value > 0.5 means that an attribute is present.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
