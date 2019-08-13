# googlenet-v3-pytorch

## Use Case and High-Level Description

Inception v3 is image classification model pretrained on ImageNet dataset. This
PyTorch implementation of architecture described in the paper ["Rethinking
the Inception Architecture for Computer Vision"](https://arxiv.org/pdf/1512.00567.pdf) in
TorchVision package (see [here](https://github.com/pytorch/vision)).

The model input is a blob that consists of a single image of "1x3x299x299"
in RGB order. 

The model output is typical object classifier for the 1000 different classifications
matching with those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 11.469        |
| MParams           | 23.817        |
| Source framework  | PyTorch\*     |

## Accuracy

## Performance

## Input

### Original model

Image, name - `data`, shape - [1x3x299x299], format [BxCxHxW],
   where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `RGB`.

Mean values - [127.5, 127.5, 127.5], scale factor for each channel - 127.5

### Converted model

Image, name - `data`, shape - [1x3x299x299], format [BxCxHxW],
   where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `BGR`.

## Output

Object classifier according to ImageNet classes, name - `prob`, shape - [1,1000] in [BxC] format, where:

- `B` - batch size
- `C` - vector of probabilities for each class in [0, 1] range

## Legal Information

[LICENSE](https://raw.githubusercontent.com/pytorch/vision/master/LICENSE)
