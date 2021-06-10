# densenet-161-tf

## Use Case and High-Level Description

This is a TensorFlow\* version of `densenet-161` model, one of the DenseNet
group of models designed to perform image classification. The weights were converted from DenseNet-Keras Models. For details see [repository](https://github.com/pudae/tensorflow-densenet/), [paper](https://arxiv.org/abs/1608.06993).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 14.128                                    |
| MParams                         | 28.666                                    |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value   |
| ------ | ------- |
| Top 1  | 76.446% |
| Top 5  | 93.228% |

## Input

### Original Model

Image, name: `Placeholder`, shape: `1, 224, 224, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `RGB`.
Mean values: [123.68, 116.78, 103.94], scale factor for each channel: 58.8235294

### Converted Model

Image, name: `Placeholder`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Floating point values in range [0, 1], which represent probabilities for classes in a dataset. Name: `densenet161/predictions/Reshape_1`.

### Converted Model

Floating point values in a range [0, 1], which represent probabilities for classes in a dataset. Name: `densenet161/predictions/Reshape_1/Transpose`, shape: `1, 1, 1, 1000`.

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/pudae/tensorflow-densenet/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-TF-DenseNet.txt](../licenses/APACHE-2.0-TF-DenseNet.txt).
