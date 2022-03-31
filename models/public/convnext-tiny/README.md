# convnext-tiny

## Use Case and High-Level Description

The `convnext-tiny` model is tiny version of ConvNeXt model, constructed entirely from standard ConvNet modules. ConvNeXt is accurate, efficient, scalable and very simple in design. The model is pre-trained for image classification task on the ImageNet dataset.

The model input is a blob that consists of a single image of `1, 3, 224, 224` in `RGB` order.

The model output is typical object classifier for the 1000 different classifications matching with those in the ImageNet database.

For details see [repository](https://github.com/rwightman/pytorch-image-models) and [paper](https://arxiv.org/abs/2201.03545).

## Specification

| Metric           | Value          |
| ---------------- | -------------- |
| Type             | Classification |
| GFLOPs           | 8.9419         |
| MParams          | 28.5892        |
| Source framework | PyTorch\*      |

## Accuracy

| Metric | Value  |
| ------ | -----  |
| Top 1  | 82.05% |
| Top 5  | 95.86% |

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
