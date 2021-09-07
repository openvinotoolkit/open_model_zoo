# googlenet-v4-tf

## Use Case and High-Level Description

The `googlenet-v4-tf` model is the most recent of the Inception family of models designed to perform image classification.
Like the other Inception models, the `googlenet-v4-tf` model has been pre-trained on the ImageNet image database.
For details about this family of models, check out the [paper](https://arxiv.org/abs/1602.07261), [repository](https://github.com/tensorflow/models/tree/master/research/slim).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 24.584        |
| MParams           | 42.648        |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 80.21%          | 80.21%         |
| Top 5  | 95.20%          | 95.20%         |

## Input

### Original model

Image, name - `input`, shape - `1, 299, 299, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.
Mean values - [127.5, 127.5, 127.5], scale value - 127.5

### Converted model

Image,  name - `data`, shape - `1, 3, 299, 299`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model

Object classifier according to ImageNet classes, name - `InceptionV4/Logits/Predictions`,  shape - `1, 1001`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `InceptionV4/Logits/Predictions`,  shape - `1, 1001`, output data format is `B, C`, where:

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
[Apache License, Version 2.0](https://github.com/tensorflow/models/blob/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-Models.txt](../licenses/APACHE-2.0-TF-Models.txt).

The original model uses the TF-Slim library, which is distributed under the
[Apache License, Version 2.0](https://github.com/google-research/tf-slim/blob/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TFSlim.txt](../licenses/APACHE-2.0-TFSlim.txt).
