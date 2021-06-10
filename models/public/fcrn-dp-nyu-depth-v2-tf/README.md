# fcrn-dp-nyu-depth-v2-tf

## Use Case and High-Level Description

This is a model for monocular depth estimation trained on the NYU Depth V2 dataset,
as described in the paper [Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373), where it is referred to as ResNet-UpProj.
The model input is a single color image.
The model output is an inverse depth map that is defined up to an unknown scale factor. More details can be found in the [following repository](https://github.com/iro-cp/FCRN-DepthPrediction).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Monodepth     |
| GFLOPs            | 63.5421       |
| MParams           | 34.5255       |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric                                                           | Value |
| ---------------------------------------------------------------- | ----- |
| [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) | 0.573 |
| log10                                                            | 0.055 |
| rel                                                              | 0.127 |

Accuracy numbers obtained on NUY Depth V2 dataset.
The `log10` metric is logarithmic absolute error, defined as `abs(log10(gt) - log10(pred))`,
where `gt` - ground truth depth map, `pred` - predicted depth map.
The `rel` metric is relative absolute error defined as absolute error normalized on ground truth depth map values
(`abs(gt - pred) / gt`, where `gt` - ground truth depth map, `pred` - predicted depth map).

## Input

### Original Model

Image, name - `Placeholder`, shape - `1, 228, 304, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.

### Converted Model

Image, name - `Placeholder`, shape - `1, 3, 228, 304`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Inverse depth map, name - `ConvPred/ConvPred`, shape - `1, 128, 160`, format is `B, H, W`, where:

- `B` - batch size
- `H` - height
- `W` - width

Inverse depth map is defined up to an unknown scale factor.

### Converted Model

Inverse depth map, name - `ConvPred/ConvPred`, shape - `1, 128, 160`, format is `B, H, W`, where:

- `B` - batch size
- `H` - height
- `W` - width

Inverse depth map is defined up to an unknown scale factor.

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

The original model is released under the following [license](https://raw.githubusercontent.com/iro-cp/FCRN-DepthPrediction/master/LICENSE):

```
Copyright (c) 2016, Iro Laina
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

[*] Other names and brands may be claimed as the property of others.
