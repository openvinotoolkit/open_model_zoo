# levit-128s

## Use Case and High-Level Description

The `levit-128s` model is one of the LeViT models family: a hybrid neural network for fast inference image classification. The model is pre-trained on the ImageNet dataset. LeViT-128s model is a small LeViT variant that has 128 channels on input of the transformer stage and 2, 3 and 4 number of pairs of Attention and MLP blocks at 1, 2 and 3 model stages respectively.

The model input is a blob that consists of a single image of `1, 3, 224, 224` in `RGB` order.

The model output is typical object classifier for the 1000 different classifications matching with those in the ImageNet database.

For details see [repository](https://github.com/rwightman/pytorch-image-models) and [paper](https://arxiv.org/abs/2104.01136).

## Specification

| Metric           | Value          |
| ---------------- | -------------- |
| Type             | Classification |
| GFLOPs           | 0.6177         |
| MParams          | 8.2199         |
| Source framework | PyTorch\*      |

## Accuracy

| Metric | Value  |
| ------ | -----  |
| Top 1  | 76.54% |
| Top 5  | 92.85% |

## Input

### Original model

Image, name - `image`,  shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.
Mean values - [123.675,116.28,103.53], scale values - [58.395, 57.12, 57.375].

### Converted model

Image, name - `image`,  shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `probs`,  shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in logits format

### Converted model

Object classifier according to ImageNet classes, name - `probs`,  shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in logits format

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
