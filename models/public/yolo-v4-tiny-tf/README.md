# yolo-v4-tiny-tf

## Use Case and High-Level Description

YOLO v4 Tiny is a real-time object detection model based on ["YOLOv4: Optimal Speed and Accuracy of Object Detection"](https://arxiv.org/abs/2004.10934) paper. It was implemented in Keras\* framework and converted to TensorFlow\* framework. For details see [repository](https://github.com/david8862/keras-YOLOv3-model-set). This model was pre-trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset with 80 classes.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 6.9289        |
| MParams           | 6.0535        |
| Source framework  | Keras\*       |

## Accuracy

Accuracy metrics obtained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) validation dataset for converted model.

| Metric                                                               | Value  |
| -------------------------------------------------------------------- | ------ |
| mAP                                                                  | 40.37% |
| [COCO mAP (0.5)](http://cocodataset.org/#detection-eval)             | 46.36% |
| [COCO mAP (0.5:0.05:0.95)](http://cocodataset.org/#detection-eval)   | 22.66% |

## Input

### Original model

Image, name - `input_1`, shape - `1, 416, 416, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Scale value - 255.

### Converted model

Image, name - `input_1`, shape - `1, 416, 416, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original model

1. The array of detection summary info, name - `conv2d_20/BiasAdd`, shape - `1, 26, 26, 255`. The anchor values are `23,27, 37,58, 81,82`.

2. The array of detection summary info, name - `conv2d_17/BiasAdd`, shape - `1, 13, 13, 255`. The anchor values are `81,82, 135,169, 344,319`.

For each case format is `B, Cx, Cy, N*85,`, where:

- `B` - batch size
- `Cx`, `Cy` - cell index
- `N` - number of detection boxes for cell

Detection box has format [`x`, `y`, `h`, `w`, `box_score`, `class_no_1`, ..., `class_no_80`], where:

- (`x`, `y`) - raw coordinates of box center, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get relative to the cell coordinates
- `h`, `w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get absolute height and width values
- `box_score` - confidence of detection box, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence in [0, 1] range
- `class_no_1`, ..., `class_no_80` - probability distribution over the classes in logits format, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) and multiply by obtained confidence value to get confidence of each class

The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of object. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file.

### Converted model

1. The array of detection summary info, name - `conv2d_20/BiasAdd/Add`, shape - `1, 26, 26, 255`. The anchor values are `23,27, 37,58, 81,82`.

2. The array of detection summary info, name - `conv2d_17/BiasAdd/Add`, shape - `1, 13, 13, 255`. The anchor values are `81,82, 135,169, 344,319`.

For each case format is `B, Cx, Cy, N*85`, where:

- `B` - batch size
- `N` - number of detection boxes for cell
- `Cx`, `Cy` - cell index

Detection box has format [`x`, `y`, `h`, `w`, `box_score`, `class_no_1`, ..., `class_no_80`], where:

- (`x`, `y`) - raw coordinates of box center, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get relative to the cell coordinates
- `h`, `w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get absolute height and width values
- `box_score` - confidence of detection box, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence in [0, 1] range
- `class_no_1`, ..., `class_no_80` - probability distribution over the classes in logits format, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) and multiply by obtained confidence value to get confidence of each class

The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of object. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file.

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
[license](https://raw.githubusercontent.com/david8862/keras-YOLOv3-model-set/master/LICENSE):

```
MIT License

Copyright (c) 2019 david8862

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
