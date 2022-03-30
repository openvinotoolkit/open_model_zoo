# nanodet-m-1.5x-416

## Use Case and High-Level Description

The `nanodet-m-1.5x-416` model is one from NanoDet models family, which is a FCOS-style one-stage anchor-free object detection model which using Generalized Focal Loss as classification and regression loss. The model is a super-fast and high accuracy lightweight model with ShuffleNetV2 1.5x backbone. This model was pre-trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset.

More details provided in the [repository](https://github.com/RangiLyu/nanodet).

## Specification

| Metric                          | Value             |
|---------------------------------|-------------------|
| Type                            | Object detection  |
| GFLOPs                          | 2.3895            |
| MParams                         | 2.0534            |
| Source framework                | PyTorch\*         |

## Accuracy

Accuracy metrics obtained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) validation dataset for converted model. Label map with 80 public available object categories are used.

| Metric              | Value  |
| ------------------- | ------ |
| coco_orig_precision | 27.38% |
| coco_precision      | 26.63% |

## Input

### Original model

Image, name - `data`, shape - `1, 3, 416, 416`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

Mean values - [103.53, 116.28, 123.675].
Scale values - [57.375, 57.12, 58.395].

### Converted model

Image, name - `data`, shape - `1, 3, 416, 416`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Output

### Original model

The array of detection summary info, name - `output`, shape - `1, 3549, 112`, format is `B, N, 112`, where:

- `B` - batch size
- `N` - number of detection boxes

Detection box has the following format:

- 80 probability distribution over the classes in logits format for 80 public available [Common Objects in Context (COCO)](https://cocodataset.org/#home) object classes, listed in file `<omz_dir>/data/dataset_classes/coco_80cl.txt`.
- 8 * 4 raw coordinates in format `A` * 4, where `A` - max value of integral set.

### Converted model

The array of detection summary info, name - `output`, shape - `1, 3549, 112`, format is `B, N, 112`, where:

- `B` - batch size
- `N` - number of detection boxes

Detection box has the following format:

- 80 probability distribution over the classes in logits format for 80 public available [Common Objects in Context (COCO)](https://cocodataset.org/#home) object classes, listed in file `<omz_dir>/data/dataset_classes/coco_80cl.txt`.
- 8 * 4 raw coordinates in format `A` * 4, where `A` - max value of integral set.

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

* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/RangiLyu/nanodet/main/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-PyTorch-NanoDet.txt`.
