# mobilenet-yolo-v4-syg

## Use Case and High-Level Description

  This is a Keras\* version of `mobilenet-yolov4` model designed to perform real-time vehicle detection.
  The weights are pretrained by BDD100k and retrained by our own dataset.
  For details, see [the repository](https://github.com/legendary111/mobilenet-yolo-v4-syg/),
  [paper of MobileNetV2](https://arxiv.org/abs/1801.04381) and [YOLOv4](https://arxiv.org/abs/2004.10934).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 65.984        |
| MParams           | 61.922        |
| Source framework  | Keras\*       |

## Accuracy

Accuracy metrics obtained on [SYGDate0829](https://github.com/ermubuzhiming/OMZ-files-download/releases/tag/v1-ly)'SYGDate0829.z01''SYGDate0829.z02''SYGDate0829.z03''SYGDate0829.zip'
which is our own made\* validation dataset for converted model.

| Metric |  Value |
| ------ | -------|
| mAP    | 86.35% |

## Input

### Original model

Image, name - `input_1`, shape - `1, 416, 416, 3`, format is `B, H, W, C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Scale value - 255.

### Converted model

Image, name - `input_1`, shape - `1, 416, 416, 3`, format is `B, H, W, C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original model

1. The array of detection summary info, name - `separable_conv2d_22/BiasAdd`,  shape - `1,52,52,27`. The anchor values are `12,16,  19,36,  40,28`.

2. The array of detection summary info, name - `separable_conv2d_30/BiasAdd`,  shape - `1,26,26,27`. The anchor values are `36,75,  76,55,  72,146`.

3. The array of detection summary info, name - `separable_conv2d_38/BiasAdd`,  shape - `1,13,13,27`. The anchor values are `142,110,  192,243,  459,401`.

For each case format is `B,Cx,Cy,N*14,`, where
    - `B` - batch size
    - `Cx`, `Cy` - cell index
    - `N` - number of detection boxes for cell

Detection box has format [`x`,`y`,`h`,`w`,`box_score`,`class_no_1`, ..., `class_no_4`], where:
- (`x`,`y`) - raw coordinates of box center, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get relative to the cell coordinates
- `h`,`w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get absolute height and width values
- `box_score` - confidence of detection box, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence in [0,1] range
- `class_no_1`,...,`class_no_4` - probability distribution over the classes in logits format, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) and multiply by obtained confidence value to get confidence of each class

### Converted model

1. The array of detection summary info, name - `separable_conv2d_22/separable_conv2d/YoloRegion`,  shape - `1, 52, 52, 27`. The anchor values are `12,16,  19,36,  40,28`.

2. The array of detection summary info, name - `separable_conv2d_30/separable_conv2d/YoloRegion`,  shape - `1, 26, 26, 27`. The anchor values are `36,75,  76,55,  72,146`.

3. The array of detection summary info, name - `separable_conv2d_38/separable_conv2d/YoloRegion`,  shape - `1, 13, 13, 27`. The anchor values are `142,110,  192,243,  459,401`.

Detection box has format [`x`,`y`,`h`,`w`,`box_score`,`class_no_1`, ..., `class_no_4`], where:
- (`x`,`y`) - coordinates of box center relative to the cell
- `h`,`w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get absolute height and width values
- `box_score` - confidence of detection box in [0,1] range
- `class_no_1`,...,`class_no_4` - probability distribution over the classes in the [0,1] range, multiply by confidence value to get confidence of each class

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

* [Object Detection C++ Demo](../../../demos/object_detection_demo/cpp/README.md)
* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)

## Legal Information

The original model is distributed under the following
[license1](https://raw.githubusercontent.com/david8862/keras-YOLOv3-model-set/master/LICENSE):

```
MIT License
Copyright (c) 2021 BJTU-SYG

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
