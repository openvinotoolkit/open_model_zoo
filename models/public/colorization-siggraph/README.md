# colorization-siggraph

## Use Case and High-Level Description

The `colorization-siggraph` model is one of the [colorization](https://arxiv.org/abs/1705.02999)
group of models designed to real-time user-guided image colorization. Model was trained on ImageNet dataset with synthetically generated user interaction.
For details about this family of models, check out the [repository](https://github.com/richzhang/colorization).

Model consumes as input L-channel of LAB-image (also user points and binary mask as optional inputs).
Model give as output predict A- and B-channels of LAB-image.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Colorization  |
| GFLOPs            | 150.5441      |
| MParams           | 34.0511       |
| Source framework  | PyTorch\*     |

## Accuracy

The accuracy metrics were calculated between generated images by model and real validation images from ImageNet dataset.
Results are obtained on subset of 2000 images.

| Metric | Value     |
| ------ | --------- |
| PSNR   | 27.73dB   |
| SSIM   | 0.92      |

Also, metrics can be calculated using [VGG16](https://arxiv.org/abs/1409.1556) caffe model and colorization as preprocessing.
The results below are obtained on the validation images from ImageNet dataset.

For preprocessing `rgb -> gray -> colorization` received values:

| Metric         | Value with preprocessing   | Value without preprocessing |
|----------------|----------------------------|-----------------------------|
| Accuracy top-1 |                     58.25% |                      70.96% |
| Accuracy top-5 |                     81.78% |                      89.88% |

## Input

1. Image, name - `data_l`, shape - `1, 1, 256, 256`, format is `B, C, H, W`, where:

   - `B` - batch size
   - `C` - channel
   - `H` - height
   - `W` - width

   L-channel of LAB-image.

2. Image, name - `user_ab`, shape - `1, 2, 256, 256`, format is `B, C, H, W`, where:

   - `B` - batch size
   - `C` - channel
   - `H` - height
   - `W` - width

   Channel order is AB channels of LAB-image. Input for user points.

3. Mask, name - `user_map`, shape - `1, 1, 256, 256`, format is `B, C, H, W`, where:

   - `B` - batch size
   - `C` - number of flags for pixel
   - `H` - height
   - `W` - width

   This input is a binary mask indicating which points are
   provided by the user. The mask differentiates unspecified points
   from user-specified gray points with (a,b) = 0.
   If point(pixel) was specified the flag will be equal to 1.

> **NOTE**: You don't need to specify all 3 inputs to use the model. If you don't want to use local user hints (user points), you can use only `data_l` input. In this case, the remaining inputs (`user_ab` and `user_map`) must be filled with zeros.

## Output

Image, name - `color_ab`, shape - `1, 2, 256, 256`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is AB channels of LAB-image.

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
[license](https://raw.githubusercontent.com/richzhang/colorization/master/LICENSE):

```
Copyright (c) 2016, Richard Zhang, Phillip Isola, Alexei A. Efros
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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
