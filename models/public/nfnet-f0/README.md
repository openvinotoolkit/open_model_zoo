# nfnet-f0

## Use Case and High-Level Description

NFNet F0 is one of the image classification Normalizer-Free models pre-trained on the ImageNet dataset. NFNets are Normalizer-Free ResNets in which use Adaptive Gradient Clipping (AGC), which clips gradients based on the unit-wise ratio of gradient norms to parameter norms.

F0 variant is the baseline variant with a depth pattern [1, 2, 6, 3] (indicating how many bottleneck blocks to allocate to each stage). Each subsequent variant has this depth pattern multiplied by N (where N = 1 for F0).

The model input is a blob that consists of a single image of `1, 3, 256, 256` in `RGB` order.

The model output is typical object classifier for the 1000 different classifications matching with those in the ImageNet database.

For details see [repository](https://github.com/rwightman/pytorch-image-models) and [paper](https://arxiv.org/abs/2102.06171).

## Specification

| Metric           | Value          |
| ---------------- | -------------- |
| Type             | Classification |
| GFLOPs           | 24.8053        |
| MParams          | 71.4444        |
| Source framework | PyTorch\*      |

## Accuracy

| Metric | Value  |
| ------ | -----  |
| Top 1  | 83.34% |
| Top 5  | 96.56% |

## Input

### Original model

Image, name - `image`,  shape - `1, 3, 256, 256`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.
Mean values - [123.675,116.28,103.53], scale values - [58.395, 57.12, 57.375].

### Converted model

Image, name - `image`,  shape - `1, 3, 256, 256`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `probs`,  shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `probs`,  shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/downloader/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
python3 <omz_dir>/tools/downloader/downloader.py --name <model_name>
```

An example of using the Model Converter:
```
python3 <omz_dir>/tools/downloader/converter.py --name <model_name>
```

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-PyTorch-Image-Models.txt](../licenses/APACHE-2.0-PyTorch-Image-Models.txt).
