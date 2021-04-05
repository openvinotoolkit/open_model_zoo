# pspnet-pytorch

## Use Case and High-Level Description

`pspnet-pytorch` is a semantic segmentation model, pre-trained on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset for 21 object classes, listed in `<omz_dir>/data/dataset_classes/voc_20cl_bkgr.txt` file. The model was built on [ResNetV1-50](https://arxiv.org/pdf/1812.01187.pdf) backbone and PSP segmentation head. This model is used for pixel-level prediction tasks. For details see [repository](https://github.com/open-mmlab/mmsegmentation/tree/master), [paper](https://arxiv.org/abs/1612.01105).

## Specification

| Metric            | Value                |
|-------------------|----------------------|
| Type              | Semantic segmentation|
| GFlops            | 357.1719             |
| MParams           | 46.5827              |
| Source framework  | PyTorch\*            |

## Accuracy

| Metric    | Value |
| --------- | ----- |
| mean_iou  | 70.6% |

Accuracy metrics were obtained with fixed input resolution 512x512.

## Input

### Original model

Image, name: `input.1`, shape: `1, 3, 512, 512`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values: [123.675, 116.28, 103.53], scale values: [58.395, 57.12, 57.375]

### Converted Model

Image, name: `input.1`, shape: `1, 3, 512, 512`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Integer values in a range [0, 20], which represent an index of a predicted class for each image pixel. Name: `segmentation_map`, shape: `1, 1, 512, 512` in `B, 1, H, W` format, where:

- `B` - batch size
- `H` - image height
- `W` - image width

### Converted Model

Integer values in a range [0, 20], which represent an index of a predicted class for each image pixel. Name: `segmentation_map`, shape: `1, 1, 512, 512` in `B, 1, H, W` format, where:

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/open-mmlab/mmsegmentation/master/LICENSE).
A copy of the license is provided in [APACHE-2.0-MMSegmentation-Models.txt](../licenses/APACHE-2.0-MMSegmentation-Models.txt).
