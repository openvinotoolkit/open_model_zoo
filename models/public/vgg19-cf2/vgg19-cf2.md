# vgg19-cf2

## Use Case and High-Level Description

This is an Caffe2\* version of `vgg19` model, designed to perform image classification.
This model was converted from Caffe\* to Caffe2\* fromat. 
For details see repository <https://github.com/caffe2/models/tree/master/vgg19>,
paper <https://arxiv.org/pdf/1409.1556.pdf>
## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 39.3          |
| MParams           | 143.667       |
| Source framework  | Caffe2\*        |

## Accuracy

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

[https://raw.githubusercontent.com/keras-team/keras/master/LICENSE]()
