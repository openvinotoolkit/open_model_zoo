# erfnet

## Use Case and High-Level Description

  This is a ONNX\* version of `erfnet` model designed to perform real-time lane detection on multi-lane road (maximum number of lanes - 4).
  This model is pre-trained in PyTorch\* framework and retrained by CULane.
  For details see [repository](https://github.com/Zhangxianwen2021/ERFNet),
  paper of [ERFNet](https://doi.org/10.1109/TITS.2017.2750080) and [repository](https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/ERFNet-CULane-PyTorch)

## Specification

| Metric            |        Value          |
|-------------------|-----------------------|
| Type              | Semantic segmentation |
| GFLOPs            | 11.13                 |
| MParams           | 7.87                  |
| Source framework  | PyTorch\*             |

## Accuracy

|  Metric  |  Value |
|  ------  | -------|
| mean_iou | 76.47% |

## Input

### Original model

Image, name - `input_1`, shape - `1,3,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width


Channel order is `BGR`.

### Converted model

Image, name - `input_1`, shape - `1,3,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width


Channel order is `BGR`.

## Output

### Original model
Feature map, name - `output1`, shape - `1,5,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

It can be treated as a five-channel feature map, where each channel is information of classes: background, road line1, road line2, road line3, road line4.
Road line1, road line2, road line3 and road line4 match respectively the actual lane1, lane2, lane3 and lane4 from left to right.


### Converted model

Feature map, name - `output1`, shape - `1,5,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width


## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the Model Downloader and other automation tools as shown in the examples below.
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

* [Image Segmentation C++ Demo](../../../demos/segmentation_demo/cpp/README.md)
* [Image Segmentation Python\* Demo](../../../demos/segmentation_demo/python/README.md)
## Legal Information

The original model is distributed under the following [license1](https://raw.githubusercontent.com/Zhangxianwen2021/ERFNet/main/License).

```
MIT License
Copyright (c) 2022 BJTU-SYG

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
