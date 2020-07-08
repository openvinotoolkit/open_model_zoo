# ctdet_coco_dlav0_384

## Use Case and High-Level Description

CenterNet object detection model `ctdet_coco_dlav0_384` originally trained on PyTorch\*
then converted to ONNX\* format. CenterNet models an object as a single point - the center point of its bounding box
and uses keypoint estimation to find center points and regresses to object size.
For details see [paper](https://arxiv.org/abs/1904.07850), [repository](https://github.com/xingyizhou/CenterNet/).

### Steps to Reproduce PyTorch to ONNX Conversion
Model is provided in ONNX format, which was obtained by the following steps.

1. Clone the original repository
```sh
git clone https://github.com/xingyizhou/CenterNet
cd CenterNet
```
2. Checkout the commit that the conversion was tested on:
```sh
git checkout 8ef87b4
```
3. Apply the `pytorch-onnx.patch` patch
```sh
git apply /path/to/pytorch-onnx.patch
```
4. Follow the original [installation steps](https://github.com/xingyizhou/CenterNet/blob/8ef87b4/readme/INSTALL.md)
5. Download the [pretrained weights](https://drive.google.com/file/d/18yBxWOlhTo32_swSug_HM4q3BeWgxp_N/view)
6. Run
```sh
python convert.py ctdet --load_model /path/to/downloaded/weights.pth --exp_id coco_dlav0_384 --arch dlav0_34 --input_res 384 --gpus -1
```

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 34.994                                    |
| MParams                         | 17.911                                    |
| Source framework                | PyTorch\*                                 |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| mAP    | 41.81%          | 41.61%            |

## Performance

## Input

### Original Model

Image, name: `input.1`, shape: [1x3x384x384], format: [BxCxHxW]
where:

   - B - batch size
   - C - number of channels
   - H - image height
   - W - image width

Expected color order: BGR.
   Mean values: [104.04, 113.985, 119.85], scale values: [73.695, 69.87, 70.89].

### Converted Model

Image, name: `input.1`, shape: [1x3x384x384], format: [BxCxHxW]
where:

   - B - batch size
   - C - number of channels
   - H - image height
   - W - image width

Expected color order: BGR.

## Output

1. Object center points heatmap, name: `508`. Contains predicted objects center point, for each of the 80 categories, according to MSCOCO\* dataset version with 80 categories of objects, without background label.
2. Object size output, name: `511`. Contains predicted width and height for each object.
3. Regression output, name: `514`. Contains offsets for each prediction.

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/xingyizhou/CenterNet/master/LICENSE)

```
MIT License

Copyright (c) 2019 Xingyi Zhou
All rights reserved.

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
