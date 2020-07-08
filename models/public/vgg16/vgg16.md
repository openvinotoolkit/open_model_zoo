# vgg16

## Use Case and High-Level Description

The `vgg16` model is one of the [vgg](https://arxiv.org/abs/1409.1556) models designed to perform image classification in Caffe\*format.

The model input is a blob that consists of a single image of "1x3x224x224" in BGR order. The BGR mean values need to be subtracted as follows: [103.939, 116.779, 123.68] before passing the image blob into the network.

The model output for `vgg16` is the typical object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 30.974        |
| MParams           | 138.358       |
| Source framework  | Caffe\*        |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 70.968%|
| Top 5  | 89.878%|

## Performance

## Input

### Original mode

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [103.939, 116.779, 123.68]

### Converted model

Image, name - `data`, shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`, shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `prob`, shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

## Legal Information

The original model is distributed under the
[Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode.txt).
A copy of the license is provided in [CC-BY-4.0.txt](../licenses/CC-BY-4.0.txt).
