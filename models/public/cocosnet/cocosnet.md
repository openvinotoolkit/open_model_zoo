# cocosnet (composite)

## Use Case and High-Level Description

Cross-domain correspondence network is a exemplar-based image translation composite model, consisting of correspondence and translation parts. Model was pre-trained on ADE20k dataset.
For details see [paper](https://arxiv.org/pdf/2004.05571) and [repository](https://github.com/microsoft/CoCosNet).

## Composite model specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Source framework                | PyTorch\*                                  |

## Specification for Correspondence network

The purpose of correspondence network is to establish correspondence between input image and given exemplar.
Correspondence network return warped exemplar with semantic from input.

### Inputs

1. name: "input_seg_map", shape: [1x151x256x256] - Input semantic segmentation mask (one-hot label map) in the format [BxCxHxW],
   where:
    - B - batch size
    - C - number of classes (151 for ADE20k)
    - H - mask height
    - W - mask width

2. name: "ref_image", shape: [1x3x256x256] - An reference image (exemplar) in the format [BxCxHxW],
   where:
    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

    Expected color order is BGR.

3. name: "ref_seg_map", shape: [1x151x256x256] - A mask (one-hot label map) for reference image in the format [BxCxHxW],
   where:
    - B - batch size
    - C - number of classes (151 for ADE20k)
    - H - mask height
    - W - mask width

### Outputs

1. name: "warped_exemplar", shape: [1x3x256x256] - A warped exemplar in the format [BxCxHxW],
   where:
    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

2. name: "warped_mask", shape: [1x151x64x64] - A warped mask in the format [BxCxHxW],
   where:
    - B - batch size
    - C - number of classes (151 for ADE20k)
    - H - mask height
    - W - mask width

## Specification for Translation network

Translation network generates the final output based on the warped exemplar according to the correspondence, yielding an exemplar-based translation output.

### Inputs

1. name: "warped_exemplar", shape: [1x154x256x256] - A result of concatenate warped exemplar (output 1 of correspondence model) and input image   (input 1 of correspendence model)  in the format [BxCxHxW],
   where:
    - B - batch size
    - C - number of classes and channels (151 for ADE20k + 3 for image)
    - H - mask height
    - W - mask width

### Outputs

1. name: "exemplar_based_output", shape: [1x3x256x256] - A result (generated) image based on exemplar in the format [BxCxHxW],
   where:
    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

    Output color order is RGB.

## Accuracy of CoCosNet model
Metrics was calculated between generated images by model and real validation images from ADE20k dataset.
For some GAN metrics (IS and FID) you need to use classification model as verification network.
In our case it is [Inception-V3](../googlenet-v3/googlenet-v3.md) model.

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| PSNR   | 12.99dB        | 12.93dB         |
| SSIM   | 0.34           | 0.34            |
| IS     | 13.34          | 13.35           |
| FID    | 33.27          | 33.14           |

## Legal Information

The original model is distributed under the following
[license](https://github.com/microsoft/CoCosNet/blob/master/LICENSE):

```
MIT License

Copyright (c) Microsoft Corporation.

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
SOFTWARE
```
