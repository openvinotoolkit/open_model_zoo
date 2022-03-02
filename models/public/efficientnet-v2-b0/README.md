# efficientnet-v2-b0

## Use Case and High-Level Description

The `efficientnet-v2-b0` model is a variant of the EfficientNetV2 pre-trained on ImageNet dataset for image classification task. EfficientNetV2 is a new family of convolutional networks that have faster training speed and better parameter efficiency than previous models. A combination of training-aware neural architecture search and scaling were used in the development to jointly optimize training speed and parameter efficiency.

More details provided in the [paper](https://arxiv.org/abs/2104.00298) and [repository](https://github.com/rwightman/pytorch-image-models).

## Specification

| Metric                          | Value          |
|---------------------------------|----------------|
| Type                            | Classification |
| GFlops                          | 1.4641         |
| MParams                         | 7.1094         |
| Source framework                | PyTorch\*      |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 78.36% |
| Top 5  | 94.02% |

## Input

### Original Model

Image, name: `input`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values - [123.675, 116.28, 103.53], scale values - [58.395, 57.12, 57.375].

### Converted Model

Image, name: `input`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name: `logits`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - vector of probabilities for all dataset classes in logits format

### Converted Model

Object classifier according to ImageNet classes, name: `logits`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - vector of probabilities for all dataset classes in logits format

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

* [Classification Python\* Demo](../../../demos/classification_demo/python/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-PyTorch-Image-Models.txt`.
