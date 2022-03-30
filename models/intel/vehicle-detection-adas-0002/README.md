# vehicle-detection-adas-0002

## Use Case and High-Level Description

This is a vehicle detection network based on an SSD framework with tuned MobileNet v1 as a feature extractor.

## Example

![](./assets/vehicle-detection-adas-0002.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Average Precision (AP)          | 90.6%                                     |
| Target vehicle size             | 40 x 30 pixels on Full HD image           |
| Max objects to detect           | 200                                       |
| GFlops                          | 2.798                                     |
| MParams                         | 1.079                                     |
| Source framework                | Caffe\*                                   |

For Average Precision metric description, see [The PASCAL Visual Object Classes (VOC) Challenge](https://link.springer.com/article/10.1007/s11263-009-0275-4).

Tested on a challenging internal dataset with 3000 images and 12585 vehicles to detect.

## Inputs

Image, name: `data`, shape: `1, 3, 384, 672` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Outputs

The net outputs blob with shape: `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. Each detection has the format [`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID (1 - vehicle)
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Object Detection C++ Demo](../../../demos/object_detection_demo/cpp/README.md)
* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)
* [Pedestrian Tracker C++ Demo](../../../demos/pedestrian_tracker_demo/cpp/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
