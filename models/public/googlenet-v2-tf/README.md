# googlenet-v2-tf

## Use Case and High-Level Description

The `googlenet-v2-tf` model is one of the Inception family, designed to perform image classification.
Like the other Inception models, the `googlenet-v2-tf` model has been pre-trained on the ImageNet image database.
For details about this family of models, check out the [paper](https://arxiv.org/abs/1602.07261), [repository](https://github.com/tensorflow/models/tree/master/research/slim).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 4.058         |
| MParams           | 11.185        |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 74.09%          | 74.09%         |
| Top 5  | 91.80%          | 91.80%         |

## Input

### Original model

Image, name - `input`, shape - `1, 224, 224, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Mean values - [127.5, 127.5, 127.5], scale value - 127.5

### Converted model

Image,  shape - `1, 224, 224, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`

## Output

### Original model

Object classifier according to ImageNet classes, name - `InceptionV2/Predictions/Softmax`,  shape - `1, 1001`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `InceptionV2/Predictions/Softmax`,  shape - `1, 1001`, output data format is `B, C`, where:

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
[Apache License, Version 2.0](https://github.com/tensorflow/models/blob/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-TF-Models.txt`.

The original model uses the TF-Slim library, which is distributed under the
[Apache License, Version 2.0](https://github.com/google-research/tf-slim/blob/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-TFSlim.txt`.
