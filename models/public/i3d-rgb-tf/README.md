# i3d-rgb-tf

## Use Case and High-Level Description

The `i3d-rgb-tf` is a model for video classification, based on paper ["Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"](https://arxiv.org/abs/1705.07750). This model use RGB input stream and trained on Kinetics-400 dataset. Additionally, this model has initialize values from Inception v1 model pre-trained on ImageNet dataset.

Originally redistributed as a checkpoint file, was converted to frozen graph.

## Conversion

1. Clone or download original repository:
    ```
    git clone https://github.com/deepmind/kinetics-i3d.git
    ```
1. (Optional) Checkout the commit that the conversion was tested on:
    ```
    git checkout 0667e88
    ```
1. Install prerequisites, tested with:
    ```
    tensorflow==1.11
    tensorflow-probability==0.4.0
    dm-sonnet==1.26
    ```
1. Copy `<omz_dir>/models/public/i3d-rgb-tf/freeze.py` script to root directory of original repository and run it:
    ```
    python freeze.py
    ```

## Specification

| Metric            | Value              |
|-------------------|--------------------|
| Type              | Action recognition |
| GFLOPs            | 278.981            |
| MParams           | 12.69              |
| Source framework  | TensorFlow\*       |

## Accuracy

Accuracy validations performed on validation part of [Kinetics-400](https://www.deepmind.com/research/open-source/kinetics) dataset.  Subset consists of 400 randomly chosen videos from this dataset.

| Metric | Converted Model | Converted Model (subset 400) |
| ------ | --------------- | ---------------------------- |
| Top 1  | 65.96%          | 64.83%                        |
| Top 5  | 86.01%          | 84.58%                        |

## Input

### Original Model

Video clip, name - `Placeholder`, shape - `1, 79, 224, 224, 3`, format is `B, D, H, W, C`, where:

- `B` - batch size
- `D` - duration of input clip
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`. Mean value - 127.5, scale value - 127.5.

### Converted Model

Video clip, name - `Placeholder`, shape - `1, 79, 224, 224, 3`, format is `B, D, H, W, C`, where:

- `B` - batch size
- `D` - duration of input clip
- `H` - height
- `W` - width
- `C` - channel

Channel order is `RGB`.

## Output

### Original Model

Action classifier according to [Kinetics-400](https://www.deepmind.com/research/open-source/kinetics) action classes, name - `Softmax`, shape - `1, 400`, format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in  [0, 1] range

### Converted Model

Action classifier according to [Kinetics-400](https://www.deepmind.com/research/open-source/kinetics) action classes, name - `Softmax`, shape - `1, 400`, format is `B, C`, where:

- `B` - batch size
- `C` - predicted probabilities for each class in [0, 1] range

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

* [Action Recognition Python\* Demo](../../../demos/action_recognition_demo/python/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/deepmind/kinetics-i3d/0667e889a5904b4dc122e0ded4c332f49f8df42c/LICENSE). A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0.txt`.
