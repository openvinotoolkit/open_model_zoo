# background-matting-v2

## Use Case and High-Level Description

The `background-matting-v2` model is a high-resolution background replacement technique based on
background matting (with MobileNetV2 backbone), where an additional frame of the background is
captured and used in recovering the alpha matte and the foreground layer. This model is
pre-trained in PyTorch\* framework and converted to ONNX\* format. More details provided in
the [paper](https://arxiv.org/abs/2012.07810).
For details see the [repository](https://github.com/DmitriySidnev/BackgroundMattingV2).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Background_matting                        |
| GFlops                          | 6.7419                                    |
| MParams                         | 5.052                                     |
| Source framework                | PyTorch\*                                 |

## Accuracy

Accuracy measured on the `PhotoMatte85` dataset with input resolution 1280x720.

| Metric         | Original model | Converted model |
| -------------- | -------------- | --------------- |
| Alpha MAD      | 1.26           | 1.26            |
| Alpha MSE      | 0.41           | 0.41            |
| Alpha GRAD     | 3.35           | 3.35            |
| Foreground MSE | 1.62           | 1.62            |

* Alpha MAD - mean of absolute difference for alpha.
* Alpha MSE - mean squared error for alpha.
* Alpha GRAD - spatial-gradient metric for alpha.
* Foreground MSE - mean squared error for foreground.

> **Note**: metrics were measured using the [script](https://github.com/DmitriySidnev/RobustVideoMatting/blob/master/evaluation/evaluate_hr.py).
> Following the original paper the test dataset obtained from the original one by compositing
> samples onto 5 random background images. The result dataset can be found [here](https://drive.google.com/file/d/1fjVXmP_gsWZiCsY6BdSxIPIv1tBZWIyK/view?usp=sharing).

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

## Legal Information

The original model is distributed under the
[MIT License](https://github.com/DmitriySidnev/BackgroundMattingV2/blob/master/LICENSE).
