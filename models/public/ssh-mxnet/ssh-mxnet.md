# ssh-mxnet

## Use Case and High-Level Description

SSH: Single Stage Headless Face Detector. More details provided in the [repository](https://github.com/deepinsight/mxnet-SSH) and [paper](https://arxiv.org/abs/1708.03979).

## Specification

| Metric                                                        | Value           |
|---------------------------------------------------------------|-----------------|
| Type                                                          | Object detection|
| GFLOPs                                                        | 267.0594        |
| MParams                                                       | 19.7684         |
| Source framework                                              | MXNet\*         |

## Accuracy

| Metric | Value |
| ------ | ----- |
| mAP    | 84.80%|

## Input

### Original model:

Image, name - `data`, shape - `1, 3, 640, 640`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `RGB`.
Mean values - [123.68, 116.779, 103.939]

### Converted model:

Image, name - `data`, shape - `1, 3, 640, 640`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `BGR`.

## Output

### Original model

Model outputs are floating points tensors:

1. name: `rpn_cls_prob_reshape_stride32`, shape: `1, 4, 20, 20`, format: `B, Ax2, H, W`, represents detection scores from Feature Pyramid Network (FPN) level with stride 32 for 2 classes: background and face.

2. name: `rpn_bbox_pred_stride32`,  shape: `1, 8, 20, 20`, format: `B, Ax4, H, W`, represents *detection box deltas* from Feature Pyramid Network (FPN) level with stride 32.

5. name: `rpn_cls_prob_reshape_stride16`, shape: `1, 4, 40, 40`, format: `B, Ax2, H, W`, represents detection scores from Feature Pyramid Network (FPN) level with stride 16 for 2 classes: background and face.

6. name: `rpn_bbox_pred_stride16`,  shape: `1, 8, 40, 40`, format: `B, Ax4, H, W`, represents *detection box deltas* from Feature Pyramid Network (FPN) level with stride 16.

9. name: `rpn_cls_prob_reshape_stride8`, shape: `1, 4, 80, 80`, format: `B, Ax2, H, W`, represents detection scores from Feature Pyramid Network (FPN) level with stride 8 for 2 classes: background and face.

10. name: `rpn_bbox_pred_stride8`,  shape: `1, 8, 80, 80`, format: `B, Ax4, H, W`, represents *detection box deltas* from Feature Pyramid Network (FPN) level with stride 8.

For each output format:

- `B` - batch size
- `A` - number of anchors
- `H` - feature height
- `W` - feature width

Detection box deltas have format `[dx, dy, dh, dw]`, where:

- `(dx, dy)` - regression for left-upper corner of bounding box,
- `(dh, dw)` - regression by height and width of bounding box.

### Converted model

The converted model has the same parameters as the original model.

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
[license](https://raw.githubusercontent.com/deepinsight/mxnet-SSH/master/LICENSE):

```
MIT License

Copyright (c) 2018 Deep Insight

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
