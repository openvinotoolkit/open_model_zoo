# octave-resnet-101-0.125

## Use Case and High-Level Description

The `octave-resnet-101-0.125` model is a modification of [ResNet-101](https://arxiv.org/abs/1512.03385) with Octave convolutions from [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049) with `alpha=0.125`.  Like the original model, this model is designed for image classification. For details about family of Octave Convolution models, check out the [repository](https://github.com/facebookresearch/OctConv).


## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 13.387        |
| MParams           | 44.543        |
| Source framework  | MXNet\*       |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 79.182%|
| Top 5  | 94.42%|

## Performance

## Input

A blob that consists of a single image of `1x3x224x224` in `RGB` order. Before passing the image blob into the network, subtract RGB mean values as follows: [124,117,104]. In addition, values must be divided by 0.0167.

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

The model output for `octave-resnet-101-0.125` is a typical object-classifier output for 1000 different classifications matching those in the ImageNet database.

### Original Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1,1000`, output data format is `B,C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/facebookresearch/OctConv/master/LICENSE):

```
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
