# ultra-lightweight-face-detection-slim-320

## Use Case and High-Level Description

Ultra-lightweight Face Detection slim 320 is a version of the lightweight face detection model with network backbone simplification. The model designed for edge computing devices and pre-trained on the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset with 320x240 input resolutions.

For details see [repository](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB).

## Specification

| Metric                          | Value             |
|---------------------------------|-------------------|
| Type                            | Object detection  |
| GFLOPs                          | 0.1724            |
| MParams                         | 0.2844            |
| Source framework                | PyTorch\*         |

## Accuracy

| Metric | Value |
| ------ | ----- |
| mAP   | 83.32% |

## Input

### Original model

Image, name - `input`, shape - `1, 3, 240, 320`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `RGB`.

Mean values - [127.0, 127.0, 127.0].
Scale values - [128.0, 128.0, 128.0].


### Converted model

Image, name - `input`, shape - `1, 3, 240, 320`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Output

### Original model

1. Bounding boxes, name: `boxes`, shape - `1, 4420, 4`. Presented in format `B, A, 4`, where:

    - `B` - batch size
    - `A` - number of detected anchors

    For each detection, the description has the format: [`x_min`, `y_min`, `x_max`, `y_max`], where:

    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner (coordinates are in normalized format, in range [0, 1])

2. Scores, name: `scores`, shape - `1, 4420, 2`. Contains scores for 2 classes - the first is background, the second is face.

### Converted model

1. Bounding boxes, name: `boxes`, shape - `1, 4420, 4`. Presented in format `B, A, 4`, where:

    - `B` - batch size
    - `A` - number of detected anchors

    For each detection, the description has the format: [`x_min`, `y_min`, `x_max`, `y_max`], where:

    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner (coordinates are in normalized format, in range [0, 1])

2. Scores, name: `scores`, shape - `1, 4420, 2`. Contains scores for 2 classes - the first is background, the second is face.

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

The original model is released under the following [license](https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/LICENSE):

```
MIT License

Copyright (c) 2019 linzai

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
