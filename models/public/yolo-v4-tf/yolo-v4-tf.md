# yolo-v4-tf

## Use Case and High-Level Description

YOLO v4 is a real-time object detection model implemented with Keras\* from this [repository](https://github.com/david8862/keras-YOLOv3-model-set) and converted to TensorFlow\* framework. This model was pretrained on COCO\* dataset with 80 classes.

## Conversion

1. Download or clone the official [repository](https://github.com/david8862/keras-YOLOv3-model-set) (tested on `d38c3d8` commit).
2. Use the following commands to get original model (named `yolov4` in repository) and convert it to Keras\* format (see details in the [README.md](https://github.com/david8862/keras-YOLOv3-model-set/blob/d38c3d865f7190ee9b19a30e91f2b750a31320c1/README.md)  file in the official repository):

   1. Download YOLO v4 weights:
        ```
        wget -O weights/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
        ```

   1. Convert model weights to Keras\*:
        ```
        python tools/model_converter/convert.py --yolo4_reorder cfg/yolov4.cfg weights/yolov4.weights weights/yolov4.h5
        ```
3. Convert the produced model to protobuf format.
    ```
    python tools/model_converter/keras_to_tensorflow.py --input_model weights/yolov4.h5 --output_model=weights/yolo-v4.pb
    ```


## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 128.608       |
| MParams           | 64.33         |
| Source framework  | Keras\*       |

## Accuracy

Accuracy metrics obtained on COCO\* validation dataset for converted model.

| Metric | Value |
| ------ | ------|
| mAP    | 71.17% |
| [COCO\* mAP (0.5)](http://cocodataset.org/#detection-eval) | 75.02% |
| [COCO\* mAP (0.5:0.05:0.95)](http://cocodataset.org/#detection-eval) | 49.2% |

## Input

### Original model

Image, name - `input_1`, shape - `1,608,608,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Scale value - 255.

### Converted model

Image, name - `input_1`, shape - `1,3,608,608`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

1. The array of detection summary info, name - `conv2d_93/BiasAdd`, shape - `1,76,76,255`. The anchor values are `12,16, 19,36, 40,28`.

2. The array of detection summary info, name - `conv2d_101/BiasAdd`, shape - `1,38,38,255`. The anchor values are `36,75, 76,55, 72,146`.

3. The array of detection summary info, name - `conv2d_109/BiasAdd`, shape - `1,19,19,255`. The anchor values are `142,110, 192,243, 459,401`.

For each case format is `B,Cx,Cy,N*85,`, where
    - `B` - batch size
    - `Cx`, `Cy` - cell index
    - `N` - number of detection boxes for cell

Detection box has format [`x`,`y`,`h`,`w`,`box_score`,`class_no_1`, ..., `class_no_80`], where:
- (`x`,`y`) - raw coordinates of box center, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get relative to the cell coordinates
- `h`,`w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get absolute height and width values
- `box_score` - confidence of detection box, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence in [0,1] range
- `class_no_1`,...,`class_no_80` - probability distribution over the classes in logits format, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) and multiply by obtained confidence value to get confidence of each class

### Converted model

1. The array of detection summary info, name - `conv2d_93/BiasAdd/Add`, shape - `1,76,76,255`. The anchor values are `12,16, 19,36, 40,28`.

2. The array of detection summary info, name - `conv2d_101/BiasAdd/Add`, shape - `1,38,38,255`. The anchor values are `36,75, 76,55, 72,146`.

3. The array of detection summary info, name - `conv2d_109/BiasAdd/Add`, shape - `1,19,19,255`. The anchor values are `142,110, 192,243, 459,401`.

For each case format is `B,N*85,Cx,Cy`, where
- `B` - batch size
- `N` - number of detection boxes for cell
- `Cx`, `Cy` - cell index

Detection box has format [`x`,`y`,`h`,`w`,`box_score`,`class_no_1`, ..., `class_no_80`], where:
- (`x`,`y`) - raw coordinates of box center, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get relative to the cell coordinates
- `h`,`w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get absolute height and width values
- `box_score` - confidence of detection box, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence in [0,1] range
- `class_no_1`,...,`class_no_80` - probability distribution over the classes in logits format, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) and multiply by obtained confidence value to get confidence of each class

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
