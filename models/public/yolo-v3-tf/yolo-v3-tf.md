# yolo-v3-tf

## Use Case and High-Level Description

YOLO v3 is a real-time object detection model implemented with Keras\* from this [repository](https://github.com/david8862/keras-YOLOv3-model-set) and converted to TensorFlow\*. This model was pretrained on COCO\* dataset with 80 classes.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | -        |
| MParams           | -        |
| Source framework  | TensorFlow\*  |

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

1. The array of detection summary info, name - `conv2d_58/BiasAdd`,  shape - `1,13,13,255`.

2. The array of detection summary info, name - `conv2d_66/BiasAdd`,  shape - `1,26,26,255`.

3. The array of detection summary info, name - `conv2d_74/BiasAdd`,  shape - `1,52,52,255`.

For each case format is `N,Cx,Cy,B*3,`, where
    - `N` - batch size
    - `Cx`, `Cy` - cell index
    - `B` - detection box for cell

Detection box has format [`x`,`y`,`h`,`w`,`conf`,`class_no_1`, ..., `class_no_80`], where:
- (`x`,`y`) - coordinates of box center, relative to cell
- `h`,`w` - normalized height and width of box
- `conf` - confidence of detection box
- `class_no_1`,...,`class_no_80` - score for each class

### Converted model

1. The array of detection summary info, name - `conv2d_58/BiasAdd/YoloRegion`,  shape - `1,255,13,13`.

2. The array of detection summary info, name - `conv2d_66/BiasAdd/YoloRegion`,  shape - `1,255,26,26`.

3. The array of detection summary info, name - `conv2d_74/BiasAdd/YoloRegion`,  shape - `1,255,52,52`.

For each case format is `N,B*3,Cx,Cy`, where
    - `N` - batch size
    - `B` - detection box for cell
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
