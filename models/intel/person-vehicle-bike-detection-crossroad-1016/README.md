# person-vehicle-bike-detection-crossroad-1016

## Use Case and High-Level Description

MobileNetV2 + SSD-based network is for Person/Vehicle/Bike detection in security
surveillance applications. Works in a variety of scenes and weather/lighting
conditions.

## Example

![](./assets/person-vehicle-bike-detection-crossroad-1016.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Mean Average Precision (mAP)    | 62.55%                                    |
| AP people                       | 73.63%                                    |
| AP vehicles                     | 77.84%                                    |
| AP bikes                        | 36.18%                                    |
| Max objects to detect           | 200                                       |
| GFlops                          | 3.560                                     |
| Source framework                | PyTorch\*                                 |

Average Precision (AP) is defined as an area under the
[precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve.

Validation dataset consists of 34,757 images from various scenes and includes:

| Type of object | Number of bounding boxes |
|----------------|--------------------------|
| Vehicle        | 229,503                  |
| Pedestrian     | 240,009                  |
| Non-vehicle    | 62,643                   |

Similarly, training dataset has 219,181 images with:

| Type of object | Number of bounding boxes |
|----------------|--------------------------|
| Vehicle        | 810,323                  |
| Pedestrian     | 1,114,799                |
| Non-vehicle    | 62,334                   |

## Inputs

Image, name: `input.1`, shape: `1, 3, 512, 512` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Outputs

The net outputs blob with shape: `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. Each detection has the format [`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`], where:

- `image_id` - ID of the image in the batch
- `label` - predicted class ID (0 - non-vehicle, 1 - vehicle, 2 - person)
- `conf` - confidence for the predicted class
- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Crossroad Camera C++ Demo](../../../demos/crossroad_camera_demo/cpp/README.md)
* [Single Human Pose Estimation Demo](../../../demos/single_human_pose_estimation_demo/python/README.md)

## Legal information
[*] Other names and brands may be claimed as the property of others.
