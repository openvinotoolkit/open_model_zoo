# retinaface-anti-cov

## Use Case and High-Level Description
RetinaFace-Anti-Cov is a customized one stage face detector to help people protect themselves from CovID-19. More details provided in the [paper](https://arxiv.org/abs/1905.00641) and [repository](https://github.com/deepinsight/insightface/tree/master/RetinaFaceAntiCov)

## Specification

| Metric                                                        | Value                                                |
|---------------------------------------------------------------|------------------------------------------------------|
| Type                                                          | Object detection, object attributes, facial landmarks|
| GFLOPs                                                        | 2.7781                                               |
| MParams                                                       | 0.5955                                               |
| Source framework                                              | MXNet\*                                              |

## Accuracy

| Metric | Value |
| ------ | ----- |
| mAP | 77.1531%|

## Input

### Original model:
Image, name - `data` , shape - [1x3x640x640], format [BxCxHxW], where:

- B - batch size
- C - number of channels
- H - image height
- W - image width

Expected color order - RGB.

### Converted model:
Image, name - `data` , shape - [1x3x640x640], format [BxCxHxW], where:

- B - batch size
- C - number of channels
- H - image height
- W - image width

Expected color order - BGR.

## Output

### Original model
Model outputs are floating points tensors:
1. name: `face_rpn_cls_prob_reshape_stride32`, shape: `1,4, 20, 20`, format: `[B, Ax2, H, W]`, represents detection scores from Feature Pyramid Network (FPN) level with stride 32 for 2 classes: background and face.

2. name: `face_rpn_bbox_stride32`,  shape: `1,8,20,20`, format: `[B, Ax4, H, W]`, represents *detection box deltas* from Feature Pyramid Network (FPN) level with stride 32.

3. name: `face_rpn_landmark_pred_stride32`, shape: `1,20,20,20`, format: `[B, Ax10, H, W]`, represents *facial landmarks* from Feature Pyramid Network (FPN) level with stride 32.

4. name: `face_rpn_type_prob_reshape_stride32`, shape: `1,6,20,20`, format: `[B, Ax3, H, W]`, represents *attributes score*.

5. name: `face_rpn_cls_prob_reshape_stride16`, shape: `1,4,40,40`, format: `[B, Ax2, H, W]`, represents detection scores from Feature Pyramid Network (FPN) level with stride 16 for 2 classes: background and face.

6. name: `face_rpn_bbox_stride16`,  shape: `1,8,40,40`, format: `[B, Ax4, H, W]`, represents *detection box deltas* from Feature Pyramid Network (FPN) level with stride 16.

7. name: `face_rpn_landmark_pred_stride16`, shape: `1,20,40,40`, format: `[B, Ax10, H, W]`, represents facial landmarks from Feature Pyramid Network (FPN) level with stride 16.

8. name: `face_rpn_type_prob_reshape_stride16`, shape: `1,6,40,40`, format: `[B, Ax3, H, W]`, represents *attributes score*.

9. name: `face_rpn_cls_prob_reshape_stride16`, shape: `1,4,80,80`, format: `[B, Ax2, H, W]`, represents detection scores from Feature Pyramid Network (FPN) level with stride 8 for 2 classes: background and face.

10. name: `face_rpn_bbox_stride16`,  shape: `1,8,80,80`, format: `[B, Ax4, H, W]`, represents detection box deltas from Feature Pyramid Network (FPN) level with stride 8.

11. name: `face_rpn_landmark_pred_stride16`, shape: `1,20,80,80`, format: `[B, Ax10, H, W]`, represents facial landmarks from Feature Pyramid Network (FPN) level with stride 8.

12. name: `face_rpn_type_prob_reshape_stride8`, shape: `1,6,80,80`, format: `[B, Ax3, H, W]`, represents *attributes score*.

For each output format:
- `B` - batch size
- `A` - number of anchors
- `H` - feature height
- `W` - feature width

Detection box deltas have format `[dx, dy, dh, dw]`, where:
- `(dx, dy)` - regression for left-upper corner of bounding box,
- `(dh, dw)` - regression by height and width of bounding box.

Facial landmarks have format `[x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]`, where:
- `(x1, y1)` - coordinates of left eye
- `(x2, y2)` - coordinates of rights eye
- `(x3, y3)` - coordinates of nose
- `(x4, y4)` - coordinates of left mouth corner
- `(x5, y5)` - coordinates of right mouth corner

The third element in attributes score is a mask attribute. This score determines the presence or absence of a mask on a person.

### Converted model
The converted model has the same parameters as the original model.

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/deepinsight/insightface/master/LICENSE):

```
MIT License
Copyright (c) 2018 Jiankang Deng and Jia Guo
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
