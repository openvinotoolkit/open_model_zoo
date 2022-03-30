# person-vehicle-bike-detection-2000

## Use Case and High-Level Description

This is a person, vehicle, bike detector that is based on MobileNetV2
backbone with two SSD heads from 1/16 and 1/8 scale feature maps and clustered
prior boxes for 256x256 resolution.

## Example

![](./assets/person-vehicle-bike-detection-2000.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| AP @ [ IoU=0.50:0.95 ]          | 0.1647 (internal test set)                |
| GFlops                          | 0.787                                     |
| MParams                         | 1.821                                     |
| Source framework                | PyTorch\*                                 |

Average Precision (AP) is defined as an area under
the [precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve.

## Inputs

Image, name: `image`, shape: `1, 3, 256, 256` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Outputs

The net outputs blob with shape: `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. Each detection has the format [`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID (0 - vehicle, 1 - person, 2 - bike)
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner

## Training Pipeline

The OpenVINO [Training Extensions](https://github.com/openvinotoolkit/training_extensions/blob/misc/README.md) provide a [training pipeline](https://github.com/openvinotoolkit/training_extensions/blob/misc/models/object_detection/model_templates/person-vehicle-bike-detection/readme.md), allowing to fine-tune the model on custom dataset.

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Object Detection C++ Demo](../../../demos/object_detection_demo/cpp/README.md)
* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)
* [Pedestrian Tracker C++ Demo](../../../demos/pedestrian_tracker_demo/cpp/README.md)

## Legal Information

[*] Other names and brands may be claimed as the property of others.
