# anti-spoof-mn3

## Use Case and High-Level Description

The `anti-spoof-mn3` model is an anti-spoofing binary classifier based on the [MobileNetV3](https://arxiv.org/abs/1905.02244), trained on the [CelebA-Spoof dataset](https://arxiv.org/abs/2007.12342). It's a small, light model, trained to predict whether or not a spoof RGB image given to the input. A lot of advanced techniques have been tried and selected the best suit options for the task.
For details see original [repository](https://github.com/kprokofi/light-weight-face-anti-spoofing).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 0.15                                      |
| MParams                         | 3.02                                      |
| Source framework                | PyTorch\*                                 |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| ACER   | 3.81%          | 3.81%           |

## Input

### Original Model

Image, name: `actual_input_1`, shape: `1, 3, 128, 128`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values: [151.2405, 119.5950, 107.8395], scale factor: [63.0105, 56.4570, 55.0035]

### Converted Model

Image, name: `actual_input_1`, shape: `1, 3, 128, 128`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original model

Probabilities for two classes (0 class is a real person, 1 - is a spoof image). Name: `output1` Shape: `1, 2`, format: `B, C`, where:

- `B` - batch size
- `C` - vector of probabilities.

### Converted model

Probabilities for two classes (0 class is a real person, 1 - is a spoof image). Name: `output1` Shape: `1, 2`, format: `B, C`, where:

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
[MIT License](https://raw.githubusercontent.com/kprokofi/light-weight-face-anti-spoofing/master/LICENSE).
