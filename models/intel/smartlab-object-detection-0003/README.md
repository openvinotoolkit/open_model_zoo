# smartlab-object-detection-0003

## Use Case and High-Level Description

This is a smartlab object detector that is based on YoloX for 416x416 resolution.

## Example

![](./assets/frame0001.jpg)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| AP @ [ IoU=0.50:0.95 ]          | see PASCAL per-cls AP (internal test set) |
| GFlops                          | 1.05                                      |
| MParams                         | 0.9                                       |
| Source framework                | PyTorch\*                                 |

PASCAL per-cls AP:
|    Class     |       per-cls AP         |
|--------------|--------------------------|
|  "balance",  |  0.98                    |
|  "weights",  |  0.23 omit in this model |
|  "tweezers", |  0.27 omit in this model |
|  "box",      |  0.88                    |
|  "battery",  |  0.85                    |
|  "tray",     |  0.99                    |
|  "ruler",    |  0.97                    |
|  "rider",    |  0 omit in this model    |
|  "scale",    |  0.93                    |
|  "hand"      |  0.92                    |
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
- `label` - predicted class ID (0 - balance, 1 - weights, 2 - tweezers, 3 - box, 4 - battery, 5 - tray, 6 - ruler, 7 - rider, 8 - scale, 9 - hand)

## Training Pipeline

The OpenVINO [Training Extensions](https://github.com/openvinotoolkit/training_extensions/blob/develop/README.md) provide a [training pipeline](https://github.com/openvinotoolkit/training_extensions/blob/develop/models/object_detection/model_templates/person-vehicle-bike-detection/readme.md), allowing to fine-tune the model on custom dataset.

## Legal Information

[*] Other names and brands may be claimed as the property of others.
