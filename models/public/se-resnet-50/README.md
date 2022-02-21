# se-resnet-50

## Use Case and High-Level Description

[ResNet-50 with Squeeze-and-Excitation blocks](https://arxiv.org/abs/1709.01507)

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 7.775         |
| MParams           | 28.061        |
| Source framework  | Caffe\*       |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 77.596%|
| Top 5  | 93.85% |

## Input

### Original Model

Image, name: `data`,  shape: `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values: [104.0, 117.0, 123.0].

### Converted Model

Image, name: `data`,  shape: `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in the range [0, 1]

### Converted Model

Object classifier according to ImageNet classes, name: `prob`,  shape: `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in the range [0, 1]

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/hujie-frank/SENet/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-SENet.txt`.
