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
| mean_iou | 68.41%|

## Input

### Original model

Image, name: `ImageTensor`, shape: `1, 513, 513, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `RGB`.

### Converted Model

Image, name: `mul_1/placeholder_port_1`, shape: `1, 513, 513, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

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

## Download a Model and Convert it into OpenVINO™ IR Format

You can download models and if necessary convert them into OpenVINO™ IR format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
omz_downloader --name <model_name>
```

An example of using the Model Converter:
```
omz_converter --name <model_name>
```

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Image Segmentation C++ Demo](../../../demos/segmentation_demo/cpp/README.md)
* [Image Segmentation Python\* Demo](../../../demos/segmentation_demo/python/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/tensorflow/models/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-TF-Models.txt`.
