# person-detection-retail-0002

## Use Case and High-Level Description

This is a pedestrian detector based on backbone with hyper-feature + R-FCN for the Retail scenario.

## Example

![](./assets/person-detection-retail-0002.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| AP                              | 80.14%                                    |
| Pose coverage                   | Standing upright, parallel to image plane |
| Support of occluded pedestrians | YES                                       |
| Occlusion coverage              | <50%                                      |
| Min pedestrian height           | 80 pixels (on 1080p)                      |
| Max objects to detect           | 200                                       |
| GFlops                          | 12.427                                    |
| MParams                         | 3.244                                     |
| Source framework                | Caffe\*                                   |

Average Precision (AP) is defined as an area under the
[precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve. Validation dataset consists of ~50K of images from ~100 different scenes.

## Inputs

1. Image, name: `data`, shape: `1, 3, 544, 992` in format `1, C, H, W`, where:

    - `C` - number of channels
    - `H` - image height
    - `W` - image width

    The expected channel order is `BGR`.

2. name: `im_info`, shape: `1, 6` - An image information
    [544, 992, 992/`frame_width`, 544/`frame_height`, 992/`frame_width`, 544/`frame_height`]

## Outputs

The net outputs blob with shape: `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. Each detection has the format [`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID (1 - person)
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Pedestrian Tracker C++ Demo](../../../demos/pedestrian_tracker_demo/cpp/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
