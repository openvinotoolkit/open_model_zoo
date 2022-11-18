# efficientnet-v2-s

## Use Case and High-Level Description

The `efficientnet-v2-s` model is a small variant of the EfficientNetV2 pre-trained on ImageNet-21k dataset and fine-tuned on ImageNet-1k for image classification task. EfficientNetV2 is a new family of convolutional networks that have faster training speed and better parameter efficiency than previous models. A combination of training-aware neural architecture search and scaling were used in the development to jointly optimize training speed and parameter efficiency.

More details provided in the [paper](https://arxiv.org/abs/2104.00298) and [repository](https://github.com/rwightman/pytorch-image-models).

## Specification

| Metric                          | Value          |
|---------------------------------|----------------|
| Type                            | Classification |
| GFlops                          | 16.9406        |
| MParams                         | 21.3816        |
| Source framework                | PyTorch\*      |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 84.29% |
| Top 5  | 97.26% |

## Input

### Original Model

Image, name: `input`, shape: `1, 3, 384, 384`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values - [127.5, 127.5, 127.5], scale values - [127.5, 127.5, 127.5].

### Converted Model

Image, name: `input`, shape: `1, 3, 384, 384`, format: `B, C, H, W`, where:

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
