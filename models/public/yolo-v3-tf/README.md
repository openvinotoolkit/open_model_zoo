# yolo-v3-tf

## Use Case and High-Level Description

YOLO v3 is a real-time object detection model implemented with Keras\* from this [repository](https://github.com/david8862/keras-YOLOv3-model-set) and converted to TensorFlow\* framework. This model was pre-trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset with 80 classes.

## Conversion

1. Download or clone the official [repository](https://github.com/david8862/keras-YOLOv3-model-set) (tested on `d38c3d8` commit).
2. Use the following commands to get original model (named `yolov3` in repository) and convert it to Keras\* format (see details in the [README.md](https://github.com/david8862/keras-YOLOv3-model-set/blob/d38c3d865f7190ee9b19a30e91f2b750a31320c1/README.md)  file in the official repository):

   1. Download YOLO v3 weights:
        ```
        wget -O weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
        ```

   2. Convert model weights to Keras\*:
        ```
        python tools/model_converter/convert.py cfg/yolov3.cfg weights/yolov3.weights weights/yolov3.h5
        ```
3. Convert model to protobuf:
    ```
    python tools/model_converter/keras_to_tensorflow.py --input_model weights/yolov3.h5 --output_model=weights/yolo-v3.pb
    ```

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 65.984        |
| MParams           | 61.922        |
| Source framework  | Keras\*       |

## Accuracy

Accuracy metrics obtained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) validation dataset for converted model.

| Metric                                                | Value  |
| ----------------------------------------------------- | -------|
| mAP                                                   | 62.27% |
| [COCO mAP](https://cocodataset.org/#detection-eval)   | 67.7%  |

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

Image, name - `input_1`, shape - `1, 3, 416, 416`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

1. The array of detection summary info, name - `conv2d_58/BiasAdd`,  shape - `1, 13, 13, 255`. The anchor values are `116,90,  156,198,  373,326`.

2. The array of detection summary info, name - `conv2d_66/BiasAdd`,  shape - `1, 26, 26, 255`. The anchor values are `30,61,  62,45,  59,119`.

3. The array of detection summary info, name - `conv2d_74/BiasAdd`,  shape - `1, 52, 52, 255`. The anchor values are `10,13,  16,30,  33,23`.

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

1. The array of detection summary info, name - `conv2d_58/BiasAdd/YoloRegion`,  shape - `1, 255, 13, 13`. The anchor values are `116,90,  156,198,  373,326`.

2. The array of detection summary info, name - `conv2d_66/BiasAdd/YoloRegion`,  shape - `1, 255, 26, 26`. The anchor values are `30,61,  62,45,  59,119`.

3. The array of detection summary info, name - `conv2d_74/BiasAdd/YoloRegion`,  shape - `1, 255, 52, 52`. The anchor values are `10,13,  16,30,  33,23`.

For each case format is `B, N*85, Cx, Cy`, where:

- `B` - batch size
- `N` - number of detection boxes for cell
- `Cx`, `Cy` - cell index

Detection box has format [`x`, `y`, `h`, `w`, `box_score`, `class_no_1`, ..., `class_no_80`], where:

- (`x`, `y`) - coordinates of box center relative to the cell
- `h`, `w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get absolute height and width values
- `box_score` - confidence of detection box in [0, 1] range
- `class_no_1`, ..., `class_no_80` - probability distribution over the classes in the [0, 1] range, multiply by confidence value to get confidence of each class

The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of object. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file.

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
