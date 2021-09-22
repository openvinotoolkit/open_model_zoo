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
| GFlops                          | 0.44506                                   |
| MParams                         | 5.471                                     |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Value  |
| ------ | ------ |
| Top 1  | 75.30% |
| Top 5  | 92.62% |

## Input

### Original Model

Image, name: `input_1`, shape: `1, 224, 224, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `RGB`.

### Converted Model

Image, name: `input_1`, shape: `1, 3, 224, 224`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `StatefulPartitionedCall/MobilenetV3large/Predictions/Softmax`,  shape - `1, 1000`, output data format is `B, C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

### Converted Model

The converted model has the same parameters as the original model.

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
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-TF-Models.txt`.
