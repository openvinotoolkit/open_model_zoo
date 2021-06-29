# unet-camvid-onnx-0001

## Use Case and High-Level Description

This is a U-Net model that is designed to perform semantic segmentation. The model has been trained on the CamVid dataset from scratch using PyTorch\* framework. Training used median frequency balancing for class weighing. For details about the original floating-point model, check out [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

The model input is a blob that consists of a single image of `1, 3, 368, 480` in the `BGR` order. The pixel values are integers in the [0, 255] range.

The model output for `unet-camvid-onnx-0001` is the per-pixel probabilities of each input pixel belonging to one of the 12 classes of the CamVid dataset:
- Sky
- Building
- Pole
- Road
- Pavement
- Tree
- SignSymbol
- Fence
- Vehicle
- Pedestrian
- Bike
- Unlabeled

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| GFlops            | 260.1                 |
| MParams           | 31.03                 |
| Source framework  | PyTorch\*             |

## Accuracy

The quality metrics were calculated on the CamVid validation dataset. The `unlabeled` class had been ignored during metrics calculation.

| Metric                    | Value         |
|---------------------------|---------------|
| mIoU                      |        71.95% |

- `IOU=TP/(TP+FN+FP)`, where:
  - `TP` - number of true positive pixels for given class
  - `FN` - number of false negative pixels for given class
  - `FP` - number of false positive pixels for given class

## Input

Image, shape - `1, 3, 368, 480`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

Semantic segmentation class probabilities map, shape -`1, 12, 368, 480`, output data format is `B, C, H, W`, where:

- `B` - batch size
- `C` - predicted probabilities of input pixel belonging to class `C` in  the [0, 1] range
- `H` - horizontal coordinate of the input pixel
- `W` - vertical coordinate of the input pixel

## Legal Information
[*] Other names and brands may be claimed as the property of others.
