# yolo-v3-tf

## Use Case and High-Level Description

YOLO v3 is a real-time object detection model implemented with Keras\* from this [repository](https://github.com/david8862/keras-YOLOv3-model-set) and converted to TensorFlow\* framework. This model was pretrained on COCO\* dataset with 80 classes.

## Conversion

1. Download or clone the official [repository](https://github.com/david8862/keras-YOLOv3-model-set) (tested on `ffede5` commit).
2. Follow the instructions in README.md file in that repository to get original model and convert it to Keras\* format.
3. Convert the produced model to protobuf format.

    1. Get conversion script from [repository](https://github.com/amir-abdi/keras_to_tensorflow):
        ```buildoutcfg
        git clone https://github.com/amir-abdi/keras_to_tensorflow
        ```
    1. (Optional) Checkout the commit that the conversion was tested on:
        ```
        git checkout c841508a88faa5aa1ffc7a4947c3809ea4ec1228
        ```
    1. Apply `keras_to_tensorflow.py.patch`:
        ```
        git apply keras_to_tensorflow.py.patch
        ```
    1. Run script:
        ```
        python keras_to_tensorflow.py --input_model=<model_in>.h5 --output_model=<model_out>.pb
        ```


## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 65.984        |
| MParams           | 61.922        |
| Source framework  | Keras\*       |

## Accuracy

Accuracy metrics obtained on COCO\* validation dataset for converted model.

| Metric | Value |
| ------ | ------|
| mAP    | 62.27 |
| [COCO\* mAP](http://cocodataset.org/#detection-eval) | 67.7 |

## Input

### Original model

Image, name - `input_1`, shape - `1,416,416,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Scale value - 255.

### Converted model

Image, name - `input_1`, shape - `1,3,416,416`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

1. The array of detection summary info, name - `conv2d_58/BiasAdd`,  shape - `1,13,13,255`. The anchor values are `116,90,  156,198,  373,326`.

2. The array of detection summary info, name - `conv2d_66/BiasAdd`,  shape - `1,26,26,255`. The anchor values are `30,61,  62,45,  59,119`.

3. The array of detection summary info, name - `conv2d_74/BiasAdd`,  shape - `1,52,52,255`. The anchor values are `10,13,  16,30,  33,23`.

For each case format is `B,Cx,Cy,N*85,`, where
    - `B` - batch size
    - `Cx`, `Cy` - cell index
    - `N` - number of detection boxes for cell

Detection box has format [`x`,`y`,`h`,`w`,`conf`,`class_no_1`, ..., `class_no_80`], where:
- (`x`,`y`) - coordinates of box center, relative to cell
- `h`,`w` - normalized height and width of box
- `conf` - confidence of detection box
- `class_no_1`,...,`class_no_80` - score for each class

### Converted model

1. The array of detection summary info, name - `conv2d_58/BiasAdd/YoloRegion`,  shape - `1,255,13,13`. The anchor values are `116,90,  156,198,  373,326`.

2. The array of detection summary info, name - `conv2d_66/BiasAdd/YoloRegion`,  shape - `1,255,26,26`. The anchor values are `30,61,  62,45,  59,119`.

3. The array of detection summary info, name - `conv2d_74/BiasAdd/YoloRegion`,  shape - `1,255,52,52`. The anchor values are `10,13,  16,30,  33,23`.

For each case format is `B,N*85,Cx,Cy`, where
    - `B` - batch size
    - `N` - number of detection boxes for cell
    - `Cx`, `Cy` - cell index
Detection box has format [`x`,`y`,`h`,`w`,`conf`,`class_no_1`, ..., `class_no_80`], where:
- (`x`,`y`) - coordinates of box center, relative to cell
- `h`,`w` - normalized height and width of box
- `conf` - confidence of detection box
- `class_no_1`,...,`class_no_80` - score for each class in the [0,1] range

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
