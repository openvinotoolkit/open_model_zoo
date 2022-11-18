# detr-resnet50

## Use Case and High-Level Description

The `detr-resnet50` model is one from DEtection TRansformer (DETR) models family, which consider object detection as a direct set prediction problem. The model has ResNet50 backbone and pretrained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset for solving object detection task. DETR predicts all objects at once, and is trained end-to-end with a set loss function which performs bipartite matching between predicted and ground-truth objects. DETR simplifies the detection pipeline by dropping multiple hand-designed components that encode prior knowledge, like spatial anchors or non-maximal suppression.

More details provided in the [paper](https://arxiv.org/abs/2005.12872) and [repository](https://github.com/facebookresearch/detr).

## Specification

| Metric                          | Value             |
|---------------------------------|-------------------|
| Type                            | Object detection  |
| GFLOPs                          | 174.4708          |
| MParams                         | 41.3293           |
| Source framework                | PyTorch\*         |

## Accuracy

Accuracy metrics obtained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) validation dataset for converted model. Background label + label map with 80 public available object categories are used (original indexing to 91 categories is preserved).

| Metric              | Value  |
| ------------------- | ------ |
| coco_orig_precision | 39.27% |
| coco_precision      | 42.36% |

## Input

### Original model

Image, name - `input`, shape - `1, 3, 800, 1137`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `RGB`.

Mean values - [123.675, 116.28, 103.53].
Scale values - [58.395, 57.12, 57.375].

### Converted model

Image, name - `input`, shape - `1, 3, 800, 1137`, format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Output

### Original model

1. Bounding boxes, name: `boxes`, shape - `1, 100, 4`. Presented in format `B, A, 4`, where:

    - `B` - batch size
    - `A` - number of detected anchors

    For each detection, the description has the format: [`x`, `y`, `w`, `h`], where:

    - (`x`, `y`) - coordinates of the bounding box center(coordinates are in normalized format, in range [0, 1])
    - `w` - width of bounding box(values are in normalized format, in range [0, 1])
    - `h` - height of bounding box(values are in normalized format, in range [0, 1])

2. Scores, name: `scores`, shape - `1, 100, 92`. Contains scores in logits format for 91 [Common Objects in Context (COCO)](https://cocodataset.org/#home) object classes. The last class is `no-object` class.

### Converted model

1. Bounding boxes, name: `boxes`, shape - `1, 100, 4`. Presented in format `B, A, 4`, where:

    - `B` - batch size
    - `A` - number of detected anchors

    For each detection, the description has the format: [`x`, `y`, `w`, `h`], where:

    - (`x`, `y`) - coordinates of the bounding box center(coordinates are in normalized format, in range [0, 1])
    - `w` - width of bounding box(values are in normalized format, in range [0, 1])
    - `h` - height of bounding box(values are in normalized format, in range [0, 1])

2. Scores, name: `scores`, shape - `1, 100, 92`. Contains scores in logits format for 91 [Common Objects in Context (COCO)](https://cocodataset.org/#home) object classes. The last class is `no-object` class.

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/facebookresearch/detr/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-FacebookResearch.txt`.
