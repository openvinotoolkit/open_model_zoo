# hbonet-0.25

## Use Case and High-Level Description

The `hbonet-0.25` model is one of the classification models from [repository](https://github.com/d-li14/HBONet) with `width_mult=0.25`

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.037         |
| MParams           | 1.9300        |
| Source framework  | PyTorch\*     |

## Accuracy

| Metric | Original model |
| ------ | -------------- |
| Top 1  | 57.30%         |
| Top 5  | 79.80%         |

## Input

### Original Model

Image, name: `input`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values: [123.675, 116.28, 103.53], scale factor for each channel: [58.395, 57.12, 57.375]

### Converted Model

Image, name: `input`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

Object classifier according to ImageNet classes, shape: `1, 1000` in `B, C` format, where:

- `B` - batch size
- `C` - vector of probabilities for all dataset classes in logits format.

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/d-li14/HBONet/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0.txt`.
