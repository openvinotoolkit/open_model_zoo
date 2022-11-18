# fbcnn

## Use Case and High-Level Description

The `fbcnn` model is a flexible blind convolutional neural network to remove JPEG artifacts. Model based on ["Towards Flexible Blind JPEG Artifacts Removal"](https://arxiv.org/abs/2109.14573) paper. It was implemented in PyTorch* framework. Model works with color jpeg images. For details about this model and other jpeg artifacts removal models (for grayscale images and double jpeg restoration), check out the ["Towards Flexible Blind JPEG Artifacts Removal (FBCNN, ICCV 2021)"](https://github.com/jiaxi-jiang/FBCNN).

## Specification

| Metric           | Value           |
| ---------------- | --------------- |
| Type             | Image Processing|
| GFLOPs           | 1420.78235      |
| MParams          | 71.922          |
| Source framework | PyTorch\*       |

## Accuracy

Model was tested on [LIVE_1](https://live.ece.utexas.edu/research/quality/subjective.htm) dataset.

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| PSNR   | 34.34Db        | 34.34Db         |
| SSIM   | 0.99           | 0.99            |

## Input

### Original model

Image, name - `image_lq`,  shape - `1, 3, 512, 512`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.
Scale value - 255.

### Converted model

Image, name - `image_lq`,  shape - `1, 3, 512, 512`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original Model

Restored image, name - `image_result`,  shape - `1, 3, 512, 512`, output data format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.

### Converted Model

Restored image, name - `image_result`,  shape - `1, 3, 512, 512`, output data format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

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

* [Image Processing C++ Demo](../../../demos/image_processing_demo/cpp/README.md)

## Legal Information

The original model is distributed under the following
[license](https://raw.githubusercontent.com/jiaxi-jiang/FBCNN/main/LICENSE).
