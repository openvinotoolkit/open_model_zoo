# deeplabv3

## Use Case and High-Level Description

DeepLab is a state-of-art deep learning model for semantic image segmentation. For details see [paper](https://arxiv.org/abs/1706.05587).

## Specification

| Metric            | Value                |
|-------------------|----------------------|
| Type              | Semantic segmentation|
| GFLOPs            | 11.469               |
| MParams           | 23.819               |
| Source framework  | TensorFlow\*         |

## Accuracy

| Metric   | Value |
| -------- | ----- |
| mean_iou | 66.85%|

## Input

### Original model

Image, name: `ImageTensor`, shape: `1, 513, 513, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `RGB`.

### Converted Model

Image, name: `mul_1/placeholder_port_1`, shape: `1, 3, 513, 513`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Integer values in a range [0, 20], which represent an index of a predicted class for each image pixel. Name: `ArgMax`, shape: `1, 513, 513` in `B, H, W` format, where:

- `B` - batch size
- `H` - image height
- `W` - image width

### Converted Model

Integer values in a range [0, 20], which represent an index of a predicted class for each image pixel. Name: `ArgMax/Squeeze`, shape: `1, 513, 513` in `B, H, W` format, where:

- `B` - batch size
- `H` - image height
- `W` - image width

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
