# efficientnet-b0_auto_aug

## Use Case and High-Level Description

The `efficientnet-b0_auto_aug` model is one of the [EfficientNet](https://arxiv.org/abs/1905.11946)
models designed to perform image classification, trained with
[AutoAugmentation preprocessing](https://arxiv.org/abs/1805.09501).
This model was pre-trained in TensorFlow\*.
All the EfficientNet models have been pre-trained on the ImageNet image database.
For details about this family of models, check out the [TensorFlow Cloud TPU repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 0.819         |
| MParams           | 5.268         |
| Source framework  | TensorFlow\*  |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 76.43%          | 76.43%         |
| Top 5  | 93.04%          | 93.04%         |

## Input

### Original Model

Image, name - `image`,  shape - `1, 224, 224, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.

### Converted Model

Image, name - `sub/placeholder_port_0`,  shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `logits`,  shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in the logits format

### Converted Model

Object classifier according to ImageNet classes, name - `efficientnet-b0/model/head/dense/MatMul`,  shape - `1, 1000`, output data format is `B, C`, where:

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
