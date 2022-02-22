# swin-tiny-patch4-window7-224

## Use Case and High-Level Description

The `swin-tiny-patch4-window7-224` model is a `tiny` version of the Swin Transformer image classification models pre-trained on ImageNet dataset. Swin Transformer is Hierarchical Vision Transformer whose representation is computed with shifted windows. Each patch is treated as a token with size of 4 and its feature is set as a concatenation of the raw pixel RGB values. The model has 7 patches in each window. Stages of tiny version of model have 2, 2, 6, 2 layers respectively. Number of channels of the hidden layers in the first stage for tiny variant is 96.

More details provided in the [paper](https://arxiv.org/pdf/2103.14030.pdf) and [repository](https://github.com/rwightman/pytorch-image-models).

## Specification

| Metric                          | Value          |
|---------------------------------|----------------|
| Type                            | Classification |
| GFlops                          | 9.0280         |
| MParams                         | 28.8173        |
| Source framework                | PyTorch\*      |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 81.38% |
| Top 5  | 95.51% |

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

Object classifier according to ImageNet classes, name: `probs`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

### Converted Model

Object classifier according to ImageNet classes, name: `probs`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

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

* [Classification Benchmark C++ Demo](../../../demos/classification_benchmark_demo/cpp/README.md)
* [Classification Python\* Demo](../../../demos/classification_demo/python/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-PyTorch-Image-Models.txt`.
