# octave-se-resnet-50-0.125

## Use Case and High-Level Description

The `octave-se-resnet-50-0.125` model is a modification of `se-resnet-50` from [this paper](https://arxiv.org/pdf/1709.01507.pdf) with octave convolutions from [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049) with `alpha=0.125`. As origin, it's designed to perform image classification. For details about family of octave convolution models, check out the [repository](https://github.com/facebookresearch/OctConv).

The model input is a blob that consists of a single image of 1x3x224x224 in RGB order. The RGB mean values need to be subtracted as follows: [124,117,104] before passing the image blob into the network. In addition, values must be divided by 0.0167.

The model output for `octave-se-resnet-50-0.125` is the typical object classifier output for the 1000 different classifications matching those in the ImageNet database.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 7.246         |
| MParams           | 28.082        |
| Source framework  | MXNet\*       |

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
Mean values - [124,117,104], scale value - 59.880239521

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

[LICENSE](https://raw.githubusercontent.com/facebookresearch/OctConv/master/LICENSE)
