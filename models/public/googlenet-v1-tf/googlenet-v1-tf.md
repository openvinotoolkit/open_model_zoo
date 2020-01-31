# googlenet-v1-tf

## Use Case and High-Level Description

The `googlenet-v1-tf` model is the most recent of the Inception family of models designed to perform image classification.
Like the other Inception models, the `googlenet-v1-tf` model has been pretrained on the ImageNet image database.
For details about this family of models, check out the [paper](https://arxiv.org/pdf/1602.07261.pdf), [repository](https://github.com/tensorflow/models/tree/master/research/slim).

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 3.016         |
| MParams           | 6.619         |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 79.61          | 69.75           |
| Top 5  | 94.69          | 89.66           |

## Performance

## Input

### Original model

Image, name - `input`, shape - `1,224,224,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Mean values - [127.5, 127.5, 127.5], scale value - 127.5

### Converted model

Image,  name - `data`, shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model

Object classifier according to ImageNet classes, name - `InceptionV1/Logits/Predictions/Softmax`,  shape - `1,1001`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `InceptionV1/Logits/Predictions/Softmax`,  shape - `1,1001`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

## Legal Information

The original model is distributed under the [license](https://github.com/tensorflow/models/blob/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TensorFlow.txt](../licenses/APACHE-2.0-TensorFlow.txt).
