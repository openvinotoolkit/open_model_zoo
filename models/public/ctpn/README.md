# ctpn

## Use Case and High-Level Description

Detecting Text in Natural Image with Connectionist Text Proposal Network. For details see [paper](https://arxiv.org/abs/1609.03605).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Object detection                          |
| GFlops                          | 55.813                                    |
| MParams                         | 17.237                                    |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value |
| ------ | ----- |
| hmean  | 73.67%|

## Input

### Original Model

Image, name: `image_tensor`, shape: `1, 600, 600, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `BGR`.
Mean values: [102.9801, 115.9465, 122.7717].

### Converted Model

Image, name: `Placeholder`, shape: `1, 600, 600, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `BGR`.

## Output

### Original Model

1. Detection boxes, name: `rpn_bbox_pred/Reshape_1`, contains predicted regions, in format `B, H, W, A`, where:

    - `B` - batch size
    - `H` - image height
    - `W` - image width
    - `A` - vector of 4\*N coordinates, where N is the number of detected anchors.

2. Probability, name: `Reshape_2`, contains probabilities for predicted regions in a [0,1] range in format `B, H, W, A`, where:

    - `B` - batch size
    - `H` - image height
    - `W` - image width
    - `A` - vector of 4\*N coordinates, where N is the number of detected anchors.

### Converted Model

1. Detection boxes, name: `rpn_bbox_pred/Reshape_1`, contains predicted regions, in format `B, H, W, A`, where:

    - `B` - batch size
    - `H` - image height
    - `W` - image width
    - `A` - vector of 4\*N coordinates, where N is the number of detected anchors.

2. Probability, name: `Reshape_2`, contains probabilities for predicted regions in a [0,1] range in format `B, H, W, A`, where:

    - `B` - batch size
    - `H` - image height
    - `W` - image width
    - `A` - vector of 4\*N coordinates, where N is the number of detected anchors.

## Download a Model and Convert it into OpenVINO™ IR Format

You can download models and if necessary convert them into OpenVINO™ IR format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
omz_downloader --name <model_name>
```

An example of using the Model Converter:
```
omz_converter --name <model_name>
```

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/eragonruan/text-detection-ctpn/banjin-dev/LICENSE):

```
MIT License

Copyright (c) 2017 shaohui ruan

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
