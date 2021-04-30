# efficientnet-b5

## Use Case and High-Level Description

The `efficientnet-b5` model is one of the [EfficientNet](https://arxiv.org/abs/1905.11946)
models designed to perform image classification.
This model was pre-trained in TensorFlow\*.
All the EfficientNet models have been pre-trained on the ImageNet image database.
For details about this family of models, check out the [TensorFlow Cloud TPU repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 21.252        |
| MParams           | 30.303        |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 83.33%          | 83.33%         |
| Top 5  | 96.67%          | 96.67%         |

## Input

### Original Model

Image, name - `image`,  shape - `1, 456, 456, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.

### Converted Model

Image, name - `sub/placeholder_port_0`,  shape - `1, 456, 456, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `logits`,  shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in the logits format

### Converted Model

Object classifier according to ImageNet classes, name - `efficientnet-b5/model/head/dense/MatMul`,  shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in the logits format

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/tpu/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-TPU.txt](../licenses/APACHE-2.0-TF-TPU.txt).
