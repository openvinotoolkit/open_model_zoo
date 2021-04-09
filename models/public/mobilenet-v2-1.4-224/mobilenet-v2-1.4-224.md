# mobilenet-v2-1.4-224

## Use Case and High-Level Description

`mobilenet-v2-1.4-224` is one of MobileNets - small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models are used. For details, see the [paper](https://arxiv.org/abs/1704.04861).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 1.183                                     |
| MParams                         | 6.087                                     |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value |
| ------ | ----- |
| Top 1  | 74.09%|
| Top 5  | 91.97%|

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

Probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format. Name: `MobilenetV1/Predictions/Reshape_1`.

### Converted Model

Probabilities for all dataset classes (0 class is background). Probabilities are represented in logits format. Name: `MobilenetV1/Predictions/Softmax`, shape: `1, 1001`, format: `B, C`, where:

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
