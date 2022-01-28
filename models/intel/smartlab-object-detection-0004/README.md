# smartlab-object-detection-0004

## Use Case and High-Level Description

This is a smartlab object detector that is based on YoloX for 416x416 resolution.

## Example

![](./assets/frame0001.jpg)

## Specification


| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| AP @ [ IoU=0.50:0.95 ]          | see PASCAL per-cls AP (internal test set) |
| GFlops                          | 1.073                                     |
| MParams                         | 0.8894                                    |
| Source framework                | PyTorch\*                                 |

PASCAL per-cls AP:

|    Class     |       per-cls AP         |
|--------------|--------------------------|
|  "weights",  |    0.41                  |
|  "tweezers", |    0.48                  |
|  "battery",  |    0.80                  |

Average Precision (AP) is defined as an area under
the [precision/recall](https://en.wikipedia.org/wiki/Precision_and_recall)
curve.

## Inputs

Image, name: `images`, shape: `1, 1, 416, 416` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Outputs

The net outputs blob with shape: `1, 1, 200, 7` in the format `1, 1, N, 7`, where `N` is the number of detected
bounding boxes. Each detection has the format [`x_min`, `y_min`, `x_max`, `y_max`, `conf1`, `conf2`, `label`], where:

- (`x_min`, `y_min`) - coordinates of the top left bounding box corner
- (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner
- `conf1` - confidence for the predicted class global?
- `conf2` - confidence for the predicted class local?
- `label` - predicted class ID (0 - weights, 1 - tweezers, 2a - battery)

## Legal Information

[*] Other names and brands may be claimed as the property of others.
