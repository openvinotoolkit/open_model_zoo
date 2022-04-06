# mobilenet-v3-large-1.0-224-paddle

## Use Case and High-Level Description

`mobilenet-v3-large-1.0-224-paddle` is one of MobileNets V3 - next generation of MobileNets,
based on a combination of complementary search techniques as well as a novel architecture design.
`mobilenet-v3-large-1.0-224-paddle` is pretrained in Paddle\* framework and targeted for high resource use cases.
For details see [paper](https://arxiv.org/abs/1905.02244) and [repository](https://github.com/PaddlePaddle/PaddleClas).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 0.4565                                    |
| MParams                         | 5.468                                     |
| Source framework                | Paddle\*                                  |

## Accuracy

| Metric | Result         |
| ------ | -------------- |
| Top 1  | 75.248%        |
| Top 5  | 92.32%         |

## Input

### Original Model

Image, name: `x`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values - [123.675,116.28,103.53], scale values - [58.395, 57.12, 57.375].

### Converted Model

Image, name: `x`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `softmax_1.tmp_0`,  shape - `1, 1000`, output data format is `B, C` where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

### Converted Model

The converted model has the same parameters as the original model.

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/PaddlePaddle/PaddleClas/release/2.3/LICENSE).
