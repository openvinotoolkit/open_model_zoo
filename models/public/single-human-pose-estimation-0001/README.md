# single-human-pose-estimation-0001

## Use Case and High-Level Description

Single human pose estimation model based on [paper](https://arxiv.org/abs/1906.04104).

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| AP(coco orig)                                                 | 69.04%                  |
| GFlops                                                        | 60.125                  |
| MParams                                                       | 33.165                  |
| Source framework                                              | PyTorch\*               |

## Inputs

### Original model

Image, name: `data`, shape: `1, 3, 384, 288` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `RGB`. Mean values - [123.675, 116.28, 103.53]. Scale values - [58.395, 57.12, 57.375]

### Converted model

Image, name: `data`, shape: `1, 3, 384, 288` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Outputs

### Original model

The net outputs list of tensor. Count of list elements is 6. Every tensor with shapes: `1, 17, 48, 36` (For every keypoint own heatmap). The six outputs are necessary in order to calculate the loss in during training. But in the future, for obtaining the results of prediction and postprocessing them, the last output is used. Each following tensor gives more accurate predictions (in context metric AP).

### Converted model

The net output is a tensor with name `heatmaps` and  shape `1, 17, 48, 36`. (For every keypoint own heatmap)

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/opencv/openvino_training_extensions/develop/LICENSE).
A copy of the license is provided in [APACHE-2.0.txt](../licenses/APACHE-2.0.txt).

[*] Other names and brands may be claimed as the property of others.
