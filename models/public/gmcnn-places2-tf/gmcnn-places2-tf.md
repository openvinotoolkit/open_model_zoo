# gmcnn-places2-tf

## Use Case and High-Level Description

The `gmcnn-places2-tf` is the TensorFlow implementation of GMCNN Image Inpainting model,
aimed to estimate suitable pixel information to fill holes in images. `gmcnn-places2-tf`
is trained on Places2 dataset with free-form masks. Originally redistributed as checkpoint files,
it was converted to a frozen graph. For details see [repository](https://github.com/shepnerd/inpainting_gmcnn).

### Steps to Reproduce Conversion to Frozen Graph

1. Clone the original repository
```sh
git clone https://github.com/shepnerd/inpainting_gmcnn.git
cd inpainting_gmcnn/tensorflow
```
2. Checkout the commit that the conversion was tested on:
```sh
git checkout ba7f710
```
3. Apply `freeze_model.patch` patch
```sh
git apply path/to/freeze_model.patch
```
4. Install the [original dependencies](https://github.com/shepnerd/inpainting_gmcnn#prerequisites).
(TensorFlow\* version used - 1.14.0, CPU).
5. Download the [pretrained weights](https://drive.google.com/file/d/1aakVS0CPML_Qg-PuXGE1Xaql96hNEKOU/view?usp=sharing)
6. Run sample conversion script:
```sh
python3 freeze_model.py --ckpt_dir path/to/downloaded_weights --save_dir path/to/save_directory
```

## Example

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Image Inpainting                          |
| GFlops                          | -                                         |
| MParams                         | -                                         |
| Source framework                | TensorFlow\*                              |

## Accuracy

Accuracy metrics are obtained on 2000 image subset of VOC2012 dataset. Images were cropped to input size
and disguised at random positions with pre-generated free-form masks.

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| PSNR   | 33.47Db        | 33.47Db         |

## Performance

## Input

### Original Model

1. Image, name: `Placeholder`, shape: [1x512x680x3], format: [BxHxWxC]
  where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: BGR.

2. Mask, name: `Placeholder_1`, shape: [1x512x680x1], format: [BxHxWxC]
  where:

    - B - batch size
    - H - mask height
    - W - mask width
    - C - number of channels

### Converted Model

1. Image, name: `Placeholder`, shape: [1x3x512x680], format: [BxCxHxW]
  where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: BGR.

2. Mask, name: `Placeholder_1`, shape: [1x1x512x680], format: [BxCxHxW]
  where:

    - B - batch size
    - C - number of channels
    - H - mask height
    - W - mask width

## Output

### Original Model

Restored image, name `Cast`, shape: [1x512x680x3], format: [BxHxWxC]
  where:

    - B - batch size
    - H - image height
    - W - image width
    - C - number of channels

   Expected color order: BGR.

### Converted Model

Restored image, name: `Cast`, shape: [1x3x512x680], format: [BxCxHxW]
  where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order: BGR.

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/shepnerd/inpainting_gmcnn/master/LICENSE):

```
MIT License

Copyright (c) 2018 yiwang

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
