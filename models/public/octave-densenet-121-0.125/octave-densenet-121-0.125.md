# octave-densenet-121-0.125

## Use Case and High-Level Description

The `octave-densenet-121-0.125` model is a modification of [`densenet-121`](https://arxiv.org/pdf/1608.06993) with Octave convolutions from [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049) with `alpha=0.125`. Like the original model, this model is designed for image classification. For details about family of Octave Convolution models, check out the [repository](https://github.com/facebookresearch/OctConv).


## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 4.883         |
| MParams           | 7.977         |
| Source framework  | MXNet\*       |

## Accuracy

## Performance

## Input

A blob that consists of a single image of `1x3x224x224` in `RGB `order. Before passing the image blob into the network, subtract RGB mean values as follows: [124,117,104]. In addition, values must be divided by 0.0167.

### Original Model

Image, name: `data`,  shape: `1,3,224,224`, format: `B,C,H,W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`. 
Mean values: [124,117,104], scale value: 59.880239521.

### Converted Model

Image, name: `data`,  shape: `1,3,224,224`, format: `B,C,H,W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

The model output for `octave-densenet-121-0.125` is a typical object-classifier output for 1000 different classifications matching those in the ImageNet database.

### Original Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1,1000`, output data format is `B,C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1,1000`, output data format is `B,C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Legal Information

[LICENSE](https://raw.githubusercontent.com/facebookresearch/OctConv/master/LICENSE)
