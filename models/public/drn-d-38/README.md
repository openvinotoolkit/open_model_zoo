# drn-d-38

## Use Case and High-Level Description

The `drn-d-38` model is a one of the Dilated Residual Networks (DRN) models for semantic segmentation task. DRN models dilate ResNet models, DRN-C version additionally removes residual connections from some of the added blocks and DRN-D version is a simplified version of DRN-C.

This model pre-trained on [Cityscapes](https://www.cityscapes-dataset.com) dataset for 19 object classes, listed in `<omz_dir>/data/dataset_classes/cityscapes_19cl.txt` file. See Cityscapes classes [definition](https://www.cityscapes-dataset.com/dataset-overview) for more details.

More details provided in the [paper](https://arxiv.org/abs/1705.09914) and [repository](https://github.com/fyu/drn).

## Specification

| Metric            | Value                |
|-------------------|----------------------|
| Type              | Semantic segmentation|
| GFLOPs            | 1768.3276            |
| MParams           | 25.9939              |
| Source framework  | PyTorch\*            |

## Accuracy

| Metric    | Value |
| --------- | ----- |
| mean_iou  | 71.31%|

## Input

### Original model

Image, name: `input`, shape: `1, 3, 1024, 2048`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values: [73.975742869, 83.660769353, 73.175805779], scale values: [46.653282963, 47.574230671, 47.041147921]

### Converted Model

Image, name: `input`, shape: `1, 3, 1024, 2048`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Float values, which represent scores of a predicted class for each image pixel. The model was trained on Cityscapes dataset with 19 categories of objects. Name: `output`, shape: `1, 19, 1024, 2048` in `B, N, H, W` format, where:

- `B` - batch size
- `N` - number of classes
- `H` - image height
- `W` - image width

### Converted Model

Float values, which represent scores of a predicted class for each image pixel. The model was trained on Cityscapes dataset with 19 categories of objects. Name: `output`, shape: `1, 19, 1024, 2048` in `B, N, H, W` format, where:

- `B` - batch size
- `N` - number of classes
- `H` - image height
- `W` - image width

## Download a Model and Convert it into OpenVINO™ IR Format

You can download models and if necessary convert them into OpenVINO™ IR format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

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

The original model is distributed under the following
[license](https://raw.githubusercontent.com/fyu/drn/master/LICENSE):

```
BSD 3-Clause License

Copyright (c) 2017, Fisher Yu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
