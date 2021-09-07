# resnest-50-pytorch

## Use Case and High-Level Description

ResNeSt-50 is image classification model pre-trained on ImageNet dataset. ResNeSt is stacked in ResNet-style from modular Split-Attention blocks that enables attention across feature-map groups.

The model input is a blob that consists of a single image of `1, 3, 224, 224` in `RGB` order.

The model output is typical object classifier for the 1000 different classifications  matching with those in the ImageNet database.

For details see [repository](https://github.com/zhanghang1989/ResNeSt) and [paper](https://arxiv.org/abs/2004.08955).

## Specification

| Metric           | Value          |
| ---------------- | -------------- |
| Type             | Classification |
| GFLOPs           | 10.8148        |
| MParams          | 27.4493        |
| Source framework | PyTorch\*      |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 81.11% |
| Top 5  | 95.36% |

## Input

### Original model

Image, name - `data`,  shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.
Mean values - [123.675, 116.28, 103.53], scale values - [58.395, 57.12, 57.375].

### Converted model

Image, name - `data`,  shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

### Converted model

The converted model has the same parameters as the original model.

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/zhanghang1989/ResNeSt/master/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).

[*] Other names and brands may be claimed as the property of others.
