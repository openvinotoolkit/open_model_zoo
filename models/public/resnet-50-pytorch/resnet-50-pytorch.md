# resnet-50-pytorch

## Use Case and High-Level Description

Resnet 50 is image classification model pretrained on ImageNet dataset. This
is PyTorch implementation based on architecture described in paper ["Deep Residual
Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385) in TorchVision
package (see [here](https://github.com/pytorch/vision)).

The model input is a blob that consists of a single image of "1x3x224x224"
in RGB order. 

The model output is typical object classifier for the 1000 different classifications
matching with those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 8.216         |
| MParams           | 25.53         |
| Source framework  | PyTorch\*     |

## Accuracy

## Performance

## Input

### Original model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`. 
Mean values - [123.675,116.28,103.53], scale values - [58.395,57.12,57.375].

### Converted model

Image, name - `data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

## Legal Information

[LICENSE](https://raw.githubusercontent.com/pytorch/vision/master/LICENSE)
