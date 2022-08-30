# ERFNet

## Use Case and High-Level Description

  This is a ONNX* version of `erfnet` model designed to perform real-time lane detection.
  This model is pre-trained in PyTorch* framework and retrained by CULane.
  For details see [repository](https://github.com/Zhangxianwen2021/ERFNet),
  paper of [ERFNet](https://doi.org/10.1109/TITS.2017.2750080) and [repository](https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/ERFNet-CULane-PyTorch)

## Specification

| Metric            |        Value          |
|-------------------|-----------------------|
| Type              | Semantic segmentation |
| GFLOPs            | 11.13                 |
| MParams           | 7.87                  |
| Source framework  | PyTorch*              |

## Accuracy

|  Metric  |  Value |
|  ------  | -------|
| mean_iou | 76.47% |

## Input

### Original model

Image, name - `input_1`, shape - `1,3,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Scale value - 255.

### Converted model

Image, name - `input_1`, shape - `1,3,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original model
Image, name - `output1`, shape - `1,5,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

### Converted model

Image, name - `output1`, shape - `1,5,208,976`, format is `B,C,H,W` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the Model Downloader and other automation tools as shown in the examples below.
An example of using the Model Downloader:

python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>

An example of using the Model Converter:

python3 <omz_dir>/tools/downloader/converter.py --name <model_name>

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Lane Detection Python\* Demo](../../../demos/lane_detection_demo/python/README.md)
## Legal Information

The original model is distributed under the following [license1](https://raw.githubusercontent.com/onnx/models/master/LICENSE).

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
