# retinaface-resnet50-pytorch

## Use Case and High-Level Description

The `retinaface-resnet50-pytorch` model is a PyTorch\* implementation of medium size RetinaFace model with ResNet50 backbone for Face Localization. It can output face bounding boxes and five facial landmarks in a single forward pass. More details provided in the [paper](https://arxiv.org/abs/1905.00641) and [repository](https://github.com/biubug6/Pytorch_Retinaface)

## Specification

| Metric                                                        | Value       |
|---------------------------------------------------------------|-------------|
| AP ([WIDER](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)) | 91.78%      |
| GFLOPs                                                        | 88.8627     |
| MParams                                                       | 27.2646     |
| Source framework                                              | PyTorch\*   |

Average Precision (AP) is defined as an area under the
[precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve. All numbers were evaluated by taking into account only faces bigger than
64 x 64 pixels.

Accuracy validation approach different from described in the original [repository](https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate). In contrast to the Accuracy Checker strategy where whole set is evaluated, the validation set is divided into 3 predefined subsets(hard, medium and easy) and all subsets are verified separately in the original evaluation strategy.
For details about original WIDER results please see [repository](https://github.com/biubug6/Pytorch_Retinaface#widerface-val-performance-in-single-scale-when-using-resnet50-as-backbone-net).

## Input

### Original model:

Image, name: `data`,  shape: `1, 3, 640, 640`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values: [104.0, 117.0, 123.0].

### Converted model:

Image, name: `data`,  shape: `1, 3, 640, 640`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Model outputs are floating points tensors:

1. name: `face_rpn_cls_prob`, shape: `1, 16800, 2`, format: `B, A*C, 2`, represents detection scores for 2 classes: background and face.

2. name: `face_rpn_bbox_pred`,  shape: `1, 16800, 4`, format: `B, A*C, 4`, represents *detection box deltas*.

3. name: `face_rpn_landmark_pred`, shape: `1, 16800, 10`, format: `B, A*C, 10`, represents *facial landmarks*.

For each output format:

- `B` - batch size
- `A` - number of anchors
- `C` - sum of products of dimensions for each stride, `C = H32 * W32 + H16 * W16 + H8 * W8`
- `H` - feature height with the corresponding stride
- `W` - feature width with the corresponding stride

Detection box deltas have format `[dx, dy, dh, dw]`, where:

- `(dx, dy)` - regression for center of bounding box
- `(dh, dw)` - regression by height and width of bounding box

Facial landmarks have format `[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]`, where:

- `(x1, y1)` - coordinates of left eye
- `(x2, y2)` - coordinates of rights eye
- `(x3, y3)` - coordinates of nose
- `(x4, y4)` - coordinates of left mouth corner
- `(x5, y5)` - coordinates of right mouth corner

### Converted model

The converted model has the same outputs as the original model.

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
[license](https://raw.githubusercontent.com/biubug6/Pytorch_Retinaface/master/LICENSE.MIT):

```
MIT License

Copyright (c) 2019

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
