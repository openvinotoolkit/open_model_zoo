# yolo-v2-tf

## Use Case and High-Level Description

YOLO v2 is a real-time object detection model implemented with Keras\* from this [repository](https://github.com/david8862/keras-YOLOv3-model-set) and converted to TensorFlow\* framework. This model was pre-trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset with 80 classes.

## Conversion

1. Download or clone the original [repository](https://github.com/david8862/keras-YOLOv3-model-set) (tested on `d38c3d8` commit).
2. Use the following commands to get original model (named `yolov2` in repository) and convert it to Keras\* format (see details in the [README.md](https://github.com/david8862/keras-YOLOv3-model-set/blob/d38c3d865f7190ee9b19a30e91f2b750a31320c1/README.md)  file in the official repository):

   1. Download YOLO v2 weights:
        ```
        wget -O weights/yolov2.weights https://pjreddie.com/media/files/yolov2.weights
        ```

   2. Convert model weights to Keras\*:
        ```
        python tools/model_converter/convert.py cfg/yolov2.cfg weights/yolov2.weights weights/yolov2.h5
        ```
3. Convert model to protobuf:
    ```
    python tools/model_converter/keras_to_tensorflow.py --input_model weights/yolov2.h5 --output_model=weights/yolo-v2.pb
    ```

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 63.03         |
| MParams           | 50.95         |
| Source framework  | Keras\*       |

## Accuracy

Accuracy metrics obtained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) validation dataset for converted model.

| Metric                                                | Value  |
| ----------------------------------------------------- | -------|
| mAP                                                   | 53.15% |
| [COCO mAP](https://cocodataset.org/#detection-eval)   | 56.5%  |

## Input

### Original model

Image, name - `image_input`, shape - `1, 608, 608, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Scale value - 255.

### Converted model

Image, name - `image_input`, shape - `1, 3, 608, 608`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

The array of detection summary info, name - `conv2d_22/BiasAdd`,  shape - `1, 19, 19, 425`, format is `B, Cx, Cy, N*85`, where:

- `B` - batch size
- `Cx`, `Cy` - cell index
- `N` - number of detection boxes for cell

Detection box has format [`x`, `y`, `h`, `w`, `box_score`, `class_no_1`, ..., `class_no_80`], where:

- (`x`, `y`) - raw coordinates of box center, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get coordinates relative to the cell
- `h`, `w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get height and width values relative to the cell
- `box_score` - confidence of detection box, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence in [0, 1] range
- `class_no_1`, ..., `class_no_80` - probability distribution over the classes in logits format, apply [softmax function](https://en.wikipedia.org/wiki/Softmax_function) and multiply by obtained confidence value to get confidence of each class.

The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of object. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file.
The anchor values are `0.57273,0.677385, 1.87446,2.06253, 3.33843,5.47434, 7.88282,3.52778, 9.77052,9.16828`.

### Converted model

The array of detection summary info, name - `conv2d_22/BiasAdd/YoloRegion`,  shape - `1, 153425`, which could be reshaped to `1, 425, 19, 19` with format `B, N*85, Cx, Cy`, where:

- `B` - batch size
- `N` - number of detection boxes for cell
- `Cx`, `Cy` - cell index

Detection box has format [`x`, `y`, `h`, `w`, `box_score`, `class_no_1`, ..., `class_no_80`], where:

- (`x`, `y`) - coordinates of box center relative to the cell
- `h`, `w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply with corresponding anchors to get height and width values relative to the cell
- `box_score` - confidence of detection box in [0, 1] range
- `class_no_1`, ..., `class_no_80` - probability distribution over the classes in the [0, 1] range, multiply by confidence value to get confidence of each class

The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of object. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file.
The anchor values are `0.57273,0.677385, 1.87446,2.06253, 3.33843,5.47434, 7.88282,3.52778, 9.77052,9.16828`.

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


The MIT License (MIT)

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
