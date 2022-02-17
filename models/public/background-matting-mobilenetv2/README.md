# background-matting-mobilenetv2

## Use Case and High-Level Description

The `background-matting-mobilenetv2` model is a high-resolution background replacement technique based on
background matting (with MobileNetV2 backbone), where an additional frame of the background is
captured and used in recovering the alpha matte and the foreground layer. This model is
pre-trained in PyTorch\* framework and converted to ONNX\* format. More details provided in
the [paper](https://arxiv.org/abs/2012.07810).
For details see the [repository](https://github.com/PeterL1n/BackgroundMattingV2).
For details regarding export to ONNX see [here](https://github.com/DmitriySidnev/BackgroundMattingV2/blob/master/export_onnx.py).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Background_matting                        |
| GFlops                          | 6.7419                                    |
| MParams                         | 5.052                                     |
| Source framework                | PyTorch\*                                 |

## Accuracy

Accuracy measured on a dataset composed with foregrounds from the HumanMatting dataset and backgrounds from the OpenImagesV5 one with input resolution 1280x720.

| Metric         | Original model | Converted model |
| -------------- | -------------- | --------------- |
| Alpha MAD      | 4.32           | 4.35            |
| Alpha MSE      | 1.0            | 1.0             |
| Alpha GRAD     | 2.48           | 2.49            |
| Foreground MSE | 2.7            | 2.69            |

* Alpha MAD - mean of absolute difference for alpha.
* Alpha MSE - mean squared error for alpha.
* Alpha GRAD - spatial-gradient metric for alpha.
* Foreground MSE - mean squared error for foreground.

## Input

### Original Model

Image, name: `src`, shape: `1, 3, 720, 1280`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
scale factor: 255

Image, name: `bgr`, shape: `1, 3, 720, 1280`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
scale factor: 255

### Converted Model

Image, name: `src`, shape: `1, 3, 720, 1280`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

Image, name: `bgr`, shape: `1, 3, 720, 1280`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original model

Alpha matte. Name: `pha` Shape: `1, 1, 720, 1280`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Foreground. Name: `fgr` Shape: `1, 3, 720, 1280`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

### Converted model

Alpha matte. Name: `pha` Shape: `1, 1, 720, 1280`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Foreground. Name: `fgr` Shape: `1, 3, 720, 1280`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

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

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Background subtraction Python\* Demo](../../../demos/background_subtraction_demo/python/README.md)

## Legal Information

The original model is distributed under the
[MIT License](https://github.com/DmitriySidnev/BackgroundMattingV2/blob/master/LICENSE).
