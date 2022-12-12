# yolo-v3-tiny-onnx

## Use Case and High-Level Description

Tiny YOLO v3 is a smaller version of real-time object detection YOLO v3 model in ONNX\* format from the [repository](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3) which is converted from Keras\* model [repository](https://github.com/qqwweee/keras-yolo3) using keras2onnx [converter](https://github.com/onnx/keras-onnx). This model was pre-trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset with 80 classes.

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 5.582         |
| MParams           | 8.8509        |
| Source framework  | ONNX\*        |

## Accuracy

Accuracy metrics obtained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) validation dataset for converted model.

| Metric                                                | Value  |
| ----------------------------------------------------- | -------|
| mAP                                                   | 17.07% |
| [COCO mAP](https://cocodataset.org/#detection-eval)   | 13.64% |

## Input

### Original model

1. Image, name - `input_1`, shape - `1, 3, 416, 416`, format is `B, C, H, W`, where:

    - `B` - batch size
    - `C` - channel
    - `H` - height
    - `W` - width

    Channel order is `RGB`.
    Scale value - 255.

2. Information of input image size, name: `image_shape`, shape: `1, 2`, format: `B, C`, where:

    - `B` - batch size
    - `C` - vector of 2 values in format `H, W`, where `H` is an image height, `W` is an image width.

### Converted model

1. Image, name - `input_1`, shape - `1, 3, 416, 416`, format is `B, C, H, W`, where:

    - `B` - batch size
    - `C` - channel
    - `H` - height
    - `W` - width

    Channel order is `BGR`.

2. Information of input image size, name: `image_shape`, shape: `1, 2`, format: `B, C`, where:

    - `B` - batch size
    - `C` - vector of 2 values in format `H, W`, where `H` is an image height, `W` is an image width.

## Output

### Original model

1. Boxes coordinates, name - `yolonms_layer_1`,  shape - `1, 2535, 4`, format - `B, N, 4`, where:

    - `B` - batch size
    - `N` - number of candidates

2. Scores of boxes per class, name - `yolonms_layer_1:1`,  shape - `1, 80, 2535`, format - `B, 80, N`, where:

    - `B` - batch size
    - `N` - number of candidates

3. Selected indices from the boxes tensor, name - `yolonms_layer_1:2`,  shape - `1, 1600, 3`, format - `B, N, 3`, where:

    - `B` - batch size
    - `N` - number of detection boxes

Each index has format [`b_idx`, `cls_idx`, `box_idx`], where:

- `b_idx` - batch index
- `cls_idx` - class_index
- `box_idx`- box_index

The model was trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset version with 80 categories of object. Mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file.

### Converted model

1. Boxes coordinates, name - `yolonms_layer_1`,  shape - `1, 2535, 4`, format - `B, N, 4`, where:

    - `B` - batch size
    - `N` - number of candidates

2. Scores of boxes per class, name - `yolonms_layer_1:1`,  shape - `1, 80, 2535`, format - `B, 80, N`, where:

    - `B` - batch size
    - `N` - number of candidates

3. Selected indices from the boxes tensor, name - `yolonms_layer_1:2`,  shape - `1, 1600, 3`, format - `B, N, 3`, where:

    - `B` - batch size
    - `N` - number of detection boxes

Each index has format [`b_idx`, `cls_idx`, `box_idx`], where:

- `b_idx` - batch index
- `cls_idx` - class_index
- `box_idx`- box_index

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
[Apache License, Version 2.0](https://raw.githubusercontent.com/onnx/models/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0.txt`.
