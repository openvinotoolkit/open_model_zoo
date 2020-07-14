# mobilefacedet-v1-mxnet

## Use Case and High-Level Description

  MobileFace Detection V1 is a Light and Fast Face Detector for Edge Devices (LFFD) model based on Yolo V3 architecture and trained with MXNet\*. For details see the [repository](https://github.com/becauseofAI/MobileFace) and [paper](https://arxiv.org/pdf/1904.10633.pdf).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 3.5456       |
| MParams           | 7.6828        |
| Source framework  | MXNet\*       |

## Accuracy

| Metric | Value |
| ------ | ----- |
| mAP  | 78.7488%|

## Input

### Original model

Image, name - `data`, shape - `[1x256x256x3]`, format -`[BxHxWxC]` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Expected color order -  `BGR`.

### Converted model

The converted model has the same parameters as the original model.

> **WARNING:** Please note that the input layout of the converted model is `[BxHxWxC]`.

## Output

### Original model

1. The array of detection summary info, name - `yolov30_slice_axis1`,  shape - `1,18,8,8`. The anchor values are `118,157,  186,248,  285,379`.

2. The array of detection summary info, name - `yolov30_slice_axis2`,  shape - `1,18,16,16`. The anchor values are `43,54,  60,75,  80,106`.

3. The array of detection summary info, name - `yolov30_slice_axis3`,  shape - `1,18,32,32`. The anchor values are `10,12,  16,20,  23,29`.

For each case format is `B,N*DB,Cx,Cy`, where
    - `B` - batch size
    - `N` - number of detection boxes for cell
    - `DB` - size of each detection box
    - `Cx`, `Cy` - cell index

Detection box has format [`x`,`y`,`h`,`w`,`box_score`,`face_score`], where:
- (`x`,`y`) - raw coordinates of box center, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get coordinates relative to the cell
- `h`,`w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get height and width values relative to cell
- `box_score` - confidence of detection box, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence in [0,1] range
- `face_score` - probability that detected object belongs to face class, apply [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) to get confidence in [0,1] range

### Converted model

1. The array of detection summary info, name - `yolov30_yolooutputv30_conv0_fwd/YoloRegion`,  shape - `1,18,8,8`. The anchor values are `118,157,  186,248,  285,379`.

2. The array of detection summary info, name - `yolov30_yolooutputv31_conv0_fwd/YoloRegion`,  shape - `1,18,16,16`. The anchor values are `43,54,  60,75,  80,106`.

3. The array of detection summary info, name - `yolov30_yolooutputv32_conv0_fwd/YoloRegion`,  shape - `1,18,32,32`. The anchor values are `10,12,  16,20,  23,29`.

For each case format is `B,N*DB,Cx,Cy`, where
    - `B` - batch size
    - `N` - number of detection boxes for cell
    - `DB` - size of each detection box
    - `Cx`, `Cy` - cell index

Detection box has format [`x`,`y`,`h`,`w`,`box_score`,`face_score`], where:
- (`x`,`y`) - raw coordinates of box center to the cell
- `h`,`w` - raw height and width of box, apply [exponential function](https://en.wikipedia.org/wiki/Exponential_function) and multiply by corresponding anchors to get height and width values relative to cell
- `box_score` - confidence of detection box in [0,1] range
- `face_score` - probability that detected object belongs to face class in [0,1] range

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/becauseofAI/MobileFace/master/LICENSE):

```
MIT License

Copyright (c) 2018

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
