# se-inception

## Use Case and High-Level Description

[BN-Inception with Squeeze-and-Excitation blocks](https://arxiv.org/abs/1709.01507)

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Classification|
| GFLOPs            | 4.091         |
| MParams           | 11.922        |
| Source framework  | Caffe\*       |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 75.996%|
| Top 5  | 92.964%|

## Input

### Original model

Image, name - `data`,  shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [104.0, 117.0, 123.0].

### Converted model

Image, name - `data`,  shape - `1, 3, 224, 224`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

### Original model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1, 1000`, output data format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

### Converted model

Object classifier according to ImageNet classes, name - `prob`,  shape - `1, 1000`, output data format is `B, C`, where:

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/hujie-frank/SENet/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-SENet.txt](../licenses/APACHE-2.0-SENet.txt).
