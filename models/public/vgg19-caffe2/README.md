# vgg19-caffe2

## Use Case and High-Level Description

This is a Caffe2\* version of `vgg19` model, designed to perform image classification.
This model was converted from Caffe\* to Caffe2\* format.
For details see [repository](https://github.com/facebookarchive/models/tree/master/vgg19),
[paper](https://arxiv.org/abs/1409.1556).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 39.3          |
| MParams           | 143.667       |
| Source framework  | Caffe2\*      |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 71.062%|
| Top 5  | 89.832%|

## Input

### Original mode

Image, name - `data`,  shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [103.939, 116.779, 123.68].

### Converted model

Image, name - `data`, shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`, shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `prob`, shape - `1, 1000`, output data format is `B, C`, where:

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/facebookarchive/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).
