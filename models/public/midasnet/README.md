# midasnet

## Use Case and High-Level Description

MidasNet is a model for monocular depth estimation trained by mixing several datasets;
as described in the following paper:
[Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer](https://arxiv.org/abs/1907.01341)

The model input is a blob that consists of a single image of `1, 3, 384, 384` in `RGB` order.

The model output is an inverse depth map that is defined up to an unknown scale factor.

## Example

See [here](https://github.com/intel-isl/MiDaS)

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Monodepth     |
| GFLOPs            | 207.25144     |
| MParams           | 104.081       |
| Source framework  | PyTorch\*     |

## Accuracy

| Metric | Value   |
| ------ | ------- |
| rmse   | 0.07071 |

## Input

### Original Model

Image, name - `image`, shape - `1, 3, 384, 384`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.

Mean values - [123.675, 116.28, 103.53].
Scale values - [51.525, 50.4, 50.625].

### Converted Model

Image, name - `image`, shape - `1, 3, 384, 384`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Inverse depth map, name - `inverse_depth`, shape - `1, 384, 384`, format is `B, H, W`, where:

- `B` - batch size
- `H` - height
- `W` - width

Inverse depth map is defined up to an unknown scale factor.

### Converted Model

Inverse depth map, name - `inverse_depth`, shape - `1, 384, 384`, format is `B, H, W`, where:

- `B` - batch size
- `H` - height
- `W` - width

Inverse depth map is defined up to an unknown scale factor.

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

* [MonoDepth Python Demo](../../../demos/monodepth_demo/python/README.md)

## Legal Information

The original model is released under the following [license](https://raw.githubusercontent.com/intel-isl/MiDaS/master/LICENSE):

```
MIT License

Copyright (c) 2019 Intel ISL (Intel Intelligent Systems Lab)

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

[*] Other names and brands may be claimed as the property of others.
