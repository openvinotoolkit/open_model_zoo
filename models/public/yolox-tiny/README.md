# yolox-tiny

## Use Case and High-Level Description

The `yolox-tiny` is a tiny version of YOLOX models family for object detection tasks. YOLOX is an anchor-free version of YOLO, with a simpler design but better performance.This model was pre-trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset with 80 classes.

More details provided in the [paper](https://arxiv.org/abs/2107.08430) and [repository](https://github.com/Megvii-BaseDetection/YOLOX).

## Specification

| Metric                          | Value             |
|---------------------------------|-------------------|
| Type                            | Object detection  |
| GFLOPs                          | 6.4813            |
| MParams                         | 5.0472            |
| Source framework                | PyTorch\*         |

## Accuracy

Accuracy metrics obtained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) validation dataset for converted model.

| Metric                                                               | Value  |
| -------------------------------------------------------------------- | -------|
| mAP                                                                  | 47.85% |
| [COCO mAP (0.5)](http://cocodataset.org/#detection-eval)             | 52.56% |
| [COCO mAP (0.5:0.05:0.95)](http://cocodataset.org/#detection-eval)   | 31.82% |

## Input

### Original model

Image, name - `images`, shape - `1, 3, 416, 416`, format - `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `RGB`.

Mean values - [123.675, 116.28, 103.53].
Scale values - [58.395, 57.12, 57.375].

### Converted model

Image, name - `images`, shape - `1, 3, 416, 416`, format - `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `BGR`.

## Output

### Original model

The array of detection summary info, name - `output`,  shape - `1, 3549, 85`, format is `B, N, 85`, where:

- `B` - batch size
- `N` - number of detection boxes

Detection box has format [`x`, `y`, `h`, `w`, `box_score`, `class_no_1`, ..., `class_no_80`], where:

- (`x`, `y`) - raw coordinates of box center
- `h`, `w` - raw height and width of box
- `box_score` - confidence of detection box
- `class_no_1`, ..., `class_no_80` - probability distribution over the classes in logits format.

The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of object. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file.

### Converted model

The array of detection summary info, name - `output`,  shape - `1, 3549, 85`, format is `B, N, 85`, where:

- `B` - batch size
- `N` - number of detection boxes

Detection box has format [`x`, `y`, `h`, `w`, `box_score`, `class_no_1`, ..., `class_no_80`], where:

- (`x`, `y`) - raw coordinates of box center
- `h`, `w` - raw height and width of box
- `box_score` - confidence of detection box
- `class_no_1`, ..., `class_no_80` - probability distribution over the classes in logits format.

The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of object. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file.

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
* [Object Detection C++ Demo](../../../demos/object_detection_demo/cpp/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0-PyTorch-YOLOX.txt`.
