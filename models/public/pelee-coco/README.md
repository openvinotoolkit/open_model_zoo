# pelee-coco

## Use Case and High-Level Description

The [Pelee](https://arxiv.org/abs/1804.06882) is a Real-Time Object Detection System on Mobile Devices
based on Single Shot Detection approach. The model is implemented using the
Caffe\* framework and trained on [Common Objects in Context (COCO)](https://cocodataset.org/#home) dataset.
For details about this model, check out the [repository](https://github.com/Robert-JunWang/Pelee).

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 1,290         |
| MParams           | 5.98          |
| Source framework  | Caffe\*       |

## Accuracy

| Metric         | Value    |
| -------------- | -------- |
| coco_precision | 21.9761% |

See [here](https://github.com/Robert-JunWang/Pelee).

## Input

### Original model

Image, name - `data`, shape - `1, 3, 304, 304`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [103.94, 116.78, 123.68],  Scale - 58.8235.

### Converted model

Image, name - `data`, shape - `1, 3, 304, 304`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

The array of detection summary info, name - `detection_out`,  shape - `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID in range [1, 80], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl.txt` file
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates are in normalized format, in range [0, 1])

### Converted model

The array of detection summary info, name - `detection_out`,  shape - `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected bounding boxes. For each detection, the description has the format:
[`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID in range [1, 80], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl_bkgr.txt` file
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner (coordinates are in normalized format, in range [0, 1])
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner  (coordinates are in normalized format, in range [0, 1])

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

* [Object Detection C++ Demo](../../../demos/object_detection_demo/cpp/README.md)
* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)
* [Pedestrian Tracker C++ Demo](../../../demos/pedestrian_tracker_demo/cpp/README.md)

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://raw.githubusercontent.com/Robert-JunWang/Pelee/master/LICENSE).
A copy of the license is provided in `<omz_dir>/models/public/licenses/APACHE-2.0.txt`.

[*] Other names and brands may be claimed as the property of others.
