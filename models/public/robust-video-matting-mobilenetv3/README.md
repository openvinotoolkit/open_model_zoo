# robust-video-matting-mobilenetv3

## Use Case and High-Level Description

The `robust-video-matting-mobilenetv3` model is a robust high-resolution human video matting method that uses
a recurrent architecture to exploit temporal information in videos and achieves significant
improvements in temporal coherence and matting quality. This model is pre-trained in PyTorch\*
framework and converted to ONNX\* format. More details provided in the [paper](https://arxiv.org/abs/2108.11515).
Backbone is MobileNetV3. For details see the [repository](https://github.com/PeterL1n/RobustVideoMatting).
For details regarding export to ONNX see the [instruction](https://github.com/DmitriySidnev/RobustVideoMatting#export-to-onnx).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Background_matting                        |
| GFlops                          | 9.3892                                    |
| MParams                         | 3.7363                                    |
| Source framework                | PyTorch\*                                 |

## Accuracy

Accuracy measured on a dataset composed with foregrounds from the HumanMatting dataset and backgrounds from the OpenImagesV5 one with input resolution 1280x720.

| Metric         | Original model | Converted model |
| -------------- | -------------- | --------------- |
| Alpha MAD      | 20.79          | 20.82           |
| Alpha MSE      | 15.1           | 15.11           |
| Alpha GRAD     | 4.44           | 4.47            |
| Foreground MSE | 4.05           | 4.06            |

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

Feature map, name: `r1`, shape: `1, 16, 144, 256`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `r2`, shape: `1, 20, 72, 128`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `r3`, shape: `1, 20, 36, 64`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `r4`, shape: `1, 20, 18, 32`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

### Converted Model

Image, name: `src`, shape: `1, 3, 720, 1280`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

Feature map, name: `r1`, shape: `1, 16, 144, 256`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `r2`, shape: `1, 20, 72, 128`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `r3`, shape: `1, 20, 36, 64`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `r4`, shape: `1, 20, 18, 32`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

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

Feature map, name: `rr1`, shape: `1, 16, 144, 256`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `rr2`, shape: `1, 20, 72, 128`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `rr3`, shape: `1, 20, 36, 64`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `rr4`, shape: `1, 20, 18, 32`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

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

Feature map, name: `rr1`, shape: `1, 16, 144, 256`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `rr2`, shape: `1, 20, 72, 128`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `rr3`, shape: `1, 20, 36, 64`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

Feature map, name: `rr4`, shape: `1, 20, 18, 32`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - feature map height
- `W` - feature map width

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
[GPL-3.0 License](https://github.com/DmitriySidnev/RobustVideoMatting/blob/master/LICENSE).
