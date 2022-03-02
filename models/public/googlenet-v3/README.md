# googlenet-v3

## Use Case and High-Level Description

The `googlenet-v3` model is the first of the Inception family of models designed to perform image classification. For details about this family of models, check out the [paper](https://arxiv.org/abs/1602.07261).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 11.469        |
| MParams           | 23.819        |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Value   |
| ------ | ------- |
| Top 1  | 77.904% |
| Top 5  | 93.808% |

## Input

### Original Model

Image, name: `input`, shape: `1, 299, 299, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `RGB`.
Mean values: [127.5, 127.5, 127.5], scale factor for each channel: 127.5

### Converted Model

Image, name: `input`,  shape: `1, 299, 299, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `BGR`.

## Output

Object classifier according to ImageNet classes, name: `InceptionV3/Predictions/Softmax`, shape: `1, 1001` in `B, C` format, where:

- `B` - batch size
- `C` - vector of probabilities for all dataset classes in [0, 1] range (0 class is background).

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-TF-Models.txt`.
