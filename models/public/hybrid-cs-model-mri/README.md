# hybrid-cs-model-mri

## Use Case and High-Level Description

The `hybrid-cs-model-mri` model is a hybrid frequency-domain/image-domain deep network for Magnetic Resonance Image (MRI) reconstruction. The model is composed of a k-space network that essentially tries to fill missing k-space samples, an Inverse Discrete Fourier Transformation (IDFT) operation, and an image-domain network that acts as an anti-aliasing filter.

More details provided in the [paper](https://arxiv.org/abs/1810.12473) and [repository](https://github.com/rmsouza01/Hybrid-CS-Model-MRI).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | MRI Image Inpainting in k-Space           |
| GFlops                          | 146.6037                                  |
| MParams                         | 11.3313                                   |
| Source framework                | TensorFlow\*                              |

## Accuracy

Accuracy metrics are obtained on [Calgary-Campinas Public Brain MR Dataset](https://sites.google.com/view/calgary-campinas-dataset/home).

| Metric      | Value        |
| ----------- | ------------ |
| PSNR (mean) | 34.272884 dB |
| PSNR (std)  | 4.607115 dB  |

## Input

### Original model

MRI input, name - `input_1`, shape - `1, 2, 256, 256`, format - `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

### Converted model

MRI input, name - `input_1`, shape - `1, 2, 256, 256`, format - `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

## Output

### Original model

The net outputs a blob with the name `StatefulPartitionedCall/model/conv2d_43/BiasAdd/Add` and shape `1, 1, 256, 256`, containing reconstructed MR image.

### Converted model

The net outputs a blob with the name `StatefulPartitionedCall/model/conv2d_43/BiasAdd/Add` and shape `1, 1, 256, 256`, containing reconstructed MR image.

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
omz_downloader --name <model_name>
```

An example of using the Model Converter:
```
omz_converter --name <model_name>
```

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/rmsouza01/Hybrid-CS-Model-MRI/2ede2f96161ce70dcdc922371fe6b6b254aafcc8/LICENSE):

```
MIT License

Copyright (c) 2018 Roberto M Souza

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
