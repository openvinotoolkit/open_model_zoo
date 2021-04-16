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
1. Copy [script](./freeze.py) to root directory of original repository and run it:
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
| Top 1  | 65.96%          | 67.0%                        |
| Top 5  | 86.01%          | 88.7%                        |

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

Video clip, name - `Placeholder`, shape - `1, 79, 3, 224, 224`, format is `B, D, C, H, W`, where:

- `B` - batch size
- `D` - duration of input clip
- `C` - channel
- `H` - height
- `W` - width

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/deepmind/kinetics-i3d/0667e889a5904b4dc122e0ded4c332f49f8df42c/LICENSE). A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).
