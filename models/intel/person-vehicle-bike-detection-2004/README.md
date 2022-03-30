# person-vehicle-bike-detection-2004

## Use Case and High-Level Description

This is a person, vehicle, bike detector that is based on MobileNetV2
backbone with ATSS head for 448x256 resolution.

## Example

![](./assets/person-vehicle-bike-detection-2004.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| AP @ [ IoU=0.50:0.95 ]          | 0.274 (internal test set)                 |
| GFlops                          | 1.811                                     |
| MParams                         | 2.327                                     |
| Source framework                | PyTorch\*                                 |

Average Precision (AP) is defined as an area under
the [precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve.

## Inputs

Image, name: `image`, shape: `1, 3, 256, 448` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Outputs

1. The `boxes` is a blob with the shape `100, 5` in the format `N, 5`, where `N` is the number of detected
   bounding boxes. For each detection, the description has the format:
   [`x_min`, `y_min`, `x_max`, `y_max`, `conf`], where:

    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner
    - `conf` - confidence for the predicted class

2. The `labels` is a blob with the shape `100` in the format `N`, where `N` is the number of detected
   bounding boxes. The value of each label is equal to predicted class ID
   (0 - vehicle, 1 - person, 2 - non-vehicle).

## Training Pipeline

The OpenVINO [Training Extensions](https://github.com/openvinotoolkit/training_extensions/blob/misc/README.md) provide a [training pipeline](https://github.com/openvinotoolkit/training_extensions/blob/misc/models/object_detection/model_templates/person-vehicle-bike-detection/readme.md), allowing to fine-tune the model on custom dataset.

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Object Detection C++ Demo](../../../demos/object_detection_demo/cpp/README.md)
* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)
* [Pedestrian Tracker C++ Demo](../../../demos/pedestrian_tracker_demo/cpp/README.md)

## Legal Information

[*] Other names and brands may be claimed as the property of others.
