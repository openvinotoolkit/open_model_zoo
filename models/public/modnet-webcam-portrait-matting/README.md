# modnet-webcam-portrait-matting

## Use Case and High-Level Description

The `modnet-webcam-portrait-matting` model is a lightweight matting objective decomposition network (MODNet) for online video portrait matting in real-time with a single input image with MobileNetV2 backbone. The model is pre-trained in PyTorch\* framework and converted to ONNX\* format.

More details provided in the [paper](https://arxiv.org/abs/2011.11961) and [repository](https://github.com/ZHKKKe/MODNet).

## Specification

| Metric                          | Value              |
|---------------------------------|--------------------|
| Type                            | Background Matting |
| GFlops                          | 31.1564            |
| MParams                         | 6.4597             |
| Source framework                | PyTorch\*          |

## Accuracy

Accuracy measured on the HumanMatting dataset

| Metric   | Mean value  | Std value |
| -------- | ------------|-----------|
| MAD      | 5.66        | 6.21      |
| MSE      | 762.52      | 1494.45   |

* MAD - mean of absolute difference
* MSE - mean squared error.

## Input

### Original Model

Image, name: `input`, shape: `1, 3, 512, 512`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values - [127.5, 127.5, 127.5], scale value - 127.5.

### Converted Model

Image, name: `input`, shape: `1, 3, 512, 512`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original model

Alpha matte with values in [0, 1] range. Name: `output` Shape: `1, 1, 512, 512`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

### Converted model

Alpha matte with values in [0, 1] range. Name: `output` Shape: `1, 1, 512, 512`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
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

* [Background subtraction Python\* Demo](../../../demos/background_subtraction_demo/python/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/ZHKKKe/MODNet/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0.txt`.
