# person-detection-0106

## Use Case and High-Level Description

This is a person detector that is based on Cascade R-CNN architecture with ResNet50
backbone.

## Example

![](./assets/person-detection-0106.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| AP @ [ IoU=0.50:0.95 ]          | 0.442 (internal test set)                 |
| GFlops                          | 404.264                                   |
| MParams                         | 71.565                                    |
| Source framework                | PyTorch\*                                 |

Average Precision (AP) is defined as an area under
the [precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve.

## Inputs

Image, name: `input`, shape: `1, 3, 800, 1344` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Outputs

1. The `boxes` is a blob with the shape `100, 5` in the format `N, 5`, where `N` is the number of detected
   bounding boxes. For each detection, the description has the format
   [`x_min`, `y_min`, `x_max`, `y_max`, `conf`], where:

    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner
    - `conf` - confidence for the predicted class

2. The `labels` is a blob with the shape `100` in the format `N`, where `N` is the number of detected
   bounding boxes. It contains predicted class ID (0 - person) per each detected box.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
