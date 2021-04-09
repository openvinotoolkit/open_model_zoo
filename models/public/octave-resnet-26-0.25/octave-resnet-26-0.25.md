# octave-resnet-26-0.25

## Use Case and High-Level Description

The `octave-resnet-26-0.25` model is a modification of [`resnet-26`](https://arxiv.org/abs/1512.03385) with Octave convolutions from [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049) with `alpha=0.25`. Like the original model, this model is designed for image classification. For details about family of Octave Convolution models, check out the [repository](https://github.com/facebookresearch/OctConv).

The model input is a blob that consists of a single image of `1, 3, 224, 224` in `RGB` order. Before passing the image blob into the network, subtract RGB mean values as follows: [124, 117, 104]. In addition, values must be divided by 0.0167.

The model output for `octave-resnet-26-0.25` is a typical object-classifier output for 1000 different classifications matching those in the ImageNet database.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 3.768         |
| MParams           | 15.99         |
| Source framework  | MXNet\*       |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 76.076%|
| Top 5  | 92.584%|

## Input

### Original Model

Image, name: `data`,  shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.
Mean values: [124, 117, 104], scale value: 59.880239521.

### Converted Model

Image, name: `data`,  shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/downloader/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>
```

An example of using the Model Converter:
```
python3 <omz_dir>/tools/downloader/converter.py --name <model_name>
```

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
