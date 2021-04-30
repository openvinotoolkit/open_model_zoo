# mobilenet-v3-large-1.0-224-tf

## Use Case and High-Level Description

`mobilenet-v3-large-1.0-224-tf` is one of MobileNets V3 - next generation of MobileNets,
based on a combination of complementary search techniques as well as a novel architecture design.
`mobilenet-v3-large-1.0-224-tf` is targeted for high resource use cases.
For details see [paper](https://arxiv.org/abs/1905.02244).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 0.4536                                    |
| MParams                         | 5.4721                                    |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 75.70%          | 75.70%         |
| Top 5  | 92.76%          | 92.76%         |

## Input

### Original Model

Image, name: `input`, shape: `1, 224, 224, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `RGB`.
Mean values: [127.5, 127.5, 127.5], scale factor for each channel: 127.5

### Converted Model

Image, name: `input`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Probabilities for all dataset classes (0 class is background). Name: `MobilenetV3/Predictions/Softmax`,
shape: `1,1001`, format: `B, C`, where:

- `B` - batch size
- `C` - vector of probabilities.

### Converted Model

Probabilities for all dataset classes (0 class is background). Name: `MobilenetV3/Predictions/Softmax`,
shape: `1,1001`, format: `B, C`, where:

- `B` - batch size
- `C` - vector of probabilities.

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-Models.txt](../licenses/APACHE-2.0-TF-Models.txt).
