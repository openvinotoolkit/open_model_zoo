# resnet-50-cf2

## Use Case and High-Level Description

This is an Caffe2\* version of `resnet-50` model, designed to perform image classification.
This model was converted from Caffe\* to Caffe2\* fromat. 
For details see repository <https://github.com/caffe2/models/tree/master/resnet50>,
paper <https://arxiv.org/pdf/1512.03385.pd>

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 8.216         |
| MParams           | 25.53         |
| Source framework  | Caffe2\*       |

## Accuracy

## Performance

## Input

### Original model

Image, name - `gpu_0/data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`. 
Mean values - [103.53,116.28,123.675], scale values - [57.375,57.12,58.395]

### Converted model

Image, name - `gpu_0/data`,  shape - `1,3,224,224`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model

Object classifier according to ImageNet classes, name - `gpu_0/softmax`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `gpu_0/softmax`,  shape - `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

## Legal Information

[LICENSE](https://raw.githubusercontent.com/KaimingHe/deep-residual-networks/master/LICENSE)
