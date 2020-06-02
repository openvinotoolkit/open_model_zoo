# faceboxes-pytorch

## Use Case and High-Level Description

FaceBoxes: A CPU Real-time Face Detector with High Accuracy. For details see
the [repository](https://github.com/zisianw/FaceBoxes.PyTorch), [paper](https://arxiv.org/pdf/1708.05234.pdf)

## Example

## Specification

| Metric                          | Value                                    |
|---------------------------------|------------------------------------------|
| Type                            | Object detection                         |
| GFLOPs                          | 1.8975                                   |
| MParams                         | 1.0059                                   |
| Source framework                | PyTorch\*                                |

## Performance

## Input

### Original model

Image, name - `input.1` , shape - [1x3x1024x1024], format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.
   Mean values - [104.0, 117.0, 123.0]

### Converted model

Image, name - `input.1` , shape - [1x3x1024x1024], format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

## Output

### Original model

1. Bounding boxes deltas , name: `342`, shape - [1x21824x4]. Presented in format [BxAx4],
    where:

    - B - batch size
    - A - number of prior box anchors

2. Scores, name: `353`, shape - [1x21824x2]. Contains scores for 2 classes - the first is background, the second is face.

### Converted model

The converted model has the same parameters as the original model.

## Legal Information

The original model is distributed under the following
[license](https://github.com/zisianw/FaceBoxes.PyTorch/blob/master/LICENSE):

```
MIT License

Copyright (c) 2017 Max deGroot, Ellis Brown
Copyright (c) 2019 Zisian Wong, Shifeng Zhang

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
