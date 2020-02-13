# yolo-v2-tiny

## Use Case and High-Level Description

YOLO v2 Tiny is a real-time object detection model from TensorFlow JS\* framework. This model was pretrained on COCO\* dataset with 80 classes.

## Example

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

The array of detection summary info, name - `conv2d_9/BiasAdd`,  shape - `1,13,13,425`, format is `N,Cx,Cy,B*5` where
- `N` - batch size
- `Cx`, `Cy` - cell index
- `B` - detection box for cell

Detection box has format [`x`,`y`,`h`,`w`,`conf`,`class_no_1`, ..., `class_no_80`], where:
- (`x`,`y`) - coordinates of box center, relative to cell
- `h`,`w` - normalized height and width of box
- `conf` - confidence of detection box
- `class_no_1`,...,`class_no_80` - score for each class in logits format

### Converted model

The array of detection summary info, name - `conv2d_9/BiasAdd/YoloRegion`,  shape - `1,71825`, which could be reshaped to `1, 425, 13, 13` with format `N,B*5,Cx,Cy` where
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
[license](https://raw.githubusercontent.com/shaqian/tfjs-yolo/master/LICENSE):

```
Copyright (c) 2018 Qian Sha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
```
