# yolact-resnet50-fpn-pytorch

## Use Case and High-Level Description

YOLACT ResNet 50 is a simple, fully convolutional model for real-time instance segmentation described in "YOLACT: Real-time Instance Segmentation" [paper](https://arxiv.org/abs/1904.02689). Model pre-trained in Pytorch\* on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset.
For details, see the [repository](https://github.com/dbolya/yolact).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Instance segmentation                     |
| GFlops                          | 118.575                                   |
| MParams                         | 36.829                                    |
| Source framework                | PyTorch\*                                 |

## Accuracy

| Metric   | Value  |
| -------- | ------ |
| AP@masks | 28.00% |
| AP@boxes | 30.69% |

## Input

### Original Model

Image, name: `input.1`, shape: `1, 3, 550, 550`, format: `B, C, H, W`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `RGB`.
Mean values - [123.675, 116.78, 103.94], scale values - [58.395, 57.12, 57.375].

### Converted Model

Image, name: `input.1`, shape: `1, 3, 550, 550`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

1. Detection scores, name: `conf`. Contains score distribution over all classes in the [0,1] range . The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of objects, 0 class is for background. Output shape is `1, 19248, 81` in `B, N, C` format, where:

    - `B` - batch size,
    - `N` - number of detected boxes,
    - `C` - number of classes.

2. Detection boxes, name: `boxes`. Contains detection boxes coordinates in a format `[y_min, x_min, y_max, x_max]`, where (`x_min`, `y_min`)  are coordinates of the top left corner, (`x_max`, `y_max`) are coordinates of the right bottom corner. Coordinates are normalized in [0, 1] range. Output shape is `1, 19248, 4` in `B, N, 4` format, where:

    - `B` - batch size,
    - `N` - number of detected boxes.

3. Masks features prototypes, name: `proto`. Contains the features projection for instance mask decoding. Output shape is `1, 138, 138, 32` in `B, H, W, C`, where:

    - `B` - batch size,
    - `H` - mask height,
    - `W` - mask width,
    - `C` - channels.

4. Raw instance masks, name: `mask`. Contains segmentation heatmaps of detected objects for all classes for every output bounding box. Output shape is `B, N, C` format, where:

    - `B` - batch size,
    - `N` - number of detected boxes,
    - `C` - channels.

Final mask prediction can be obtained by matrix multiplication of `proto` and transposed `mask` outputs.

### Converted Model

Converted model outputs are the same as in the original model.

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

The original model is distributed under the
[MIT license](https://raw.githubusercontent.com/dbolya/yolact/master/LICENSE).
```
MIT License

Copyright (c) 2019 Daniel Bolya

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
