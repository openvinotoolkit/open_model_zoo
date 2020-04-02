# RetinaFace-ResNet50

## Use Case and High-Level Description
RetinaFace-R50 is a medium size model with ResNet50 backbone for Face Localisation. It can output face bounding boxes and five facial landmarks in a single forward pass. More details provided in the [paper](https://arxiv.org/abs/1905.00641) and [repository](https://github.com/deepinsight/insightface/tree/master/RetinaFace)

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| AP ([WIDER](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)) | 87.30%                  |
| GFLOPs                                                        | 100.8478                |
| MParams                                                       | 29.4276                 |
| Source framework                                              | MXNet\*                 |

Average Precision (AP) is defined as an area under the
[precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve. All numbers were evaluated by taking into account only faces bigger than
64 x 64 pixels.

Accuracy validation approach different from described in the original repo.
For details about original WIDER results please see [https://github.com/deepinsight/insightface/tree/master/RetinaFace]()

## Input

### Original model:
Image, name: `data`,  shape: `1,3,640,640`, format: `B,C,H,W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.

### Converted model:
Image, name: `data`,  shape: `1,3,640,640`, format: `B,C,H,W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model
Model outputs are floating points tensors:
1.  -
name: `face_rpn_cls_prob_reshape_stride32`, shape: `1,4, 20, 20`, format: `[B, Ax2, H, W]`, where
   - `B` - batch size
   - `A` - number of anchors
   - `H` - feature height
   - `W` - feature width
   represents detection scores from Feature Pyramid Network (FPN) level with stride 32 for 2 classes: background and face.
2.  name: `face_rpn_bbox_stride32`,  shape: `1,8,20,20`, format: `[B, Ax4, H, W]`, where
   - `B` - batch size
   - `A` - number of anchors
   - `H` - feature height
   - `W` - feature width
   represents detection box deltas from Feature Pyramid Network (FPN) level with stride 32
   Box deltas format `[dx, dy, dh, dw]`, where `(dx, dy)` - regression for left-upper corner of bounding box,
   `(dh, dw)` - regression by height and width of bounding box.
3. name: `face_rpn_landmark_pred_stride32`, shape: `1,20,20,20`, format: `[B, Ax10, H, W]`
   - `B` - batch size
   - `A` - number of anchors
   - `H` - feature height
   - `W` - feature width
   represents facial landmarks from Feature Pyramid Network (FPN) level with stride 32.
   Output landmarks format `[(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]`, where
   - `x1, y1` - coordinates of left eye
   - `x2, y2` - coordinates of rights eye
   - `x3, y3` - coordinates of nose
   - `x4, y4` - coordinates of left mouth corner
   - `x5, y5` - coordinates of right mouth corner
4. name: `face_rpn_cls_prob_reshape_stride16`, shape: `1,4,40,40`, format: `[B, Ax2, H, W]`, where
   - `B` - batch size
   - `A` - number of anchors
   - `H` - feature height
   - `W` - feature width
   represents detection scores from Feature Pyramid Network (FPN) level with stride 16 for 2 classes: background and face
5. name: `face_rpn_bbox_stride16`,  shape: `1,8,40,40`, format: `[B, Ax4, H, W]`, where
   - `B` - batch size
   - `A` - number of anchors
   - `H` - feature height
   - `W` - feature width
   represents detection box deltas from Feature Pyramid Network (FPN) level with stride 16.
   Box deltas format `[dx, dy, dh, dw]`, where `(dx, dy)` - regression for left-upper corner of bounding box,
   `(dh, dw)` - regression by height and width of bounding box.
6. name: `face_rpn_landmark_pred_stride16`, shape: `1,20,40,40`, format: `[B, Ax10, H, W]`
   - `B` - batch size
   - `A` - number of anchors
   - `H` - feature height
   - `W` - feature width
   represents facial landmarks from Feature Pyramid Network (FPN) level with stride 16.
   Output landmarks format `[(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]`, where
   - `x1, y1` - coordinates of left eye
   - `x2, y2` - coordinates of rights eye
   - `x3, y3` - coordinates of nose
   - `x4, y4` - coordinates of left mouth corner
   - `x5, y5` - coordinates of right mouth corner
7. name: `face_rpn_cls_prob_reshape_stride16`, shape: `1,4,80,80`, format: `[B, Ax2, H, W]`, where
   - `B` - batch size
   - `A` - number of anchors
   - `H` - feature height
   - `W` - feature width
   represents detection scores from Feature Pyramid Network (FPN) level with stride 8 for 2 classes: background and face.
8. name: `face_rpn_bbox_stride16`,  shape: `1,8,80,80`, format: `[B, Ax4, H, W]`, where
   - `B` - batch size
   - `A` - number of anchors
   - `H` - feature height
   - `W` - feature width
   represents detection box deltas from Feature Pyramid Network (FPN) level with stride 8.
   Box deltas format `[dx, dy, dh, dw]`, where `(dx, dy)` - regression for left-upper corner of bounding box,
   `(dh, dw)` - regression by height and width of bounding box.
9. name: `face_rpn_landmark_pred_stride16`, shape: `1,20,80,80`, format: `[B, Ax10, H, W]`
   - `B` - batch size
   - `A` - number of anchors
   - `H` - feature height
   - `W` - feature width
   represents facial landmarks from Feature Pyramid Network (FPN) level with stride 8.
   Output landmarks format `[(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]`, where
   - `x1, y1` - coordinates of left eye
   - `x2, y2` - coordinates of rights eye
   - `x3, y3` - coordinates of nose
   - `x4, y4` - coordinates of left mouth corner
   - `x5, y5` - coordinates of right mouth corner
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
