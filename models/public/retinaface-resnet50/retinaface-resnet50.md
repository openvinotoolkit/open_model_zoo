# RetinaFace-ResNet50

## Use Case and High-Level Description
## Example

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
model outputs is floating points vectors:
1. name: `face_rpn_cls_prob_reshape_stride32`, shape: `[1x4x20x20]` - Detection scores for Feature Pyramid Network (FPN) level with stride 32
2. name: `face_rpn_bbox_stride32`,  shape: `[1x8x20x20]` -  Detection boxes for Feature Pyramid Network (FPN) level with stride 32
3. name: `face_rpn_landmark_pred_stride32`, shape: `[1x20x20x20]` - Facial Landmarks for Feture Pyramid Network (FPN) level with stride 32
4. name: `face_rpn_cls_prob_reshape_stride16`, shape: `[1x4x40x40]` - Detection scores for Feature Pyramid Network (FPN) level with stride 16
5. name: `face_rpn_bbox_stride16`,  shape: `[1x8x40x40]` -  Detection boxes for Feature Pyramid Network (FPN) level with stride 16
6. name: `face_rpn_landmark_pred_stride16`, shape: `[1x20x40x40]` - Facial Landmarks for Feture Pyramid Network (FPN) level with stride 16
7. name: `face_rpn_cls_prob_reshape_stride16`, shape: `[1x4x80x80]` - Detection scores for Feature Pyramid Network (FPN) level with stride 16
8. name: `face_rpn_bbox_stride16`,  shape: `[1x8x80x80]` -  Detection boxes for Feature Pyramid Network (FPN) level with stride 16
9. name: `face_rpn_landmark_pred_stride16`, shape: `[1x20x80x80]` - Facial Landmarks for Feture Pyramid Network (FPN) level with stride 16

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
