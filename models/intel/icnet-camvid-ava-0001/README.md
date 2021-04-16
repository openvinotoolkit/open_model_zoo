# icnet-camvid-ava-0001

## Use Case and High-Level Description

A trained model of ICNet for fast semantic segmentation, trained on the CamVid dataset from scratch using the TensorFlow\* framework. For details about the original floating-point model, check out [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545).

The model input is a blob that consists of a single image of `1, 3, 720, 960` in the `BGR` order. The pixel values are integers in the [0, 255] range.

The model output for `icnet-camvid-ava-0001` is the predicted class index of each input pixel belonging to one of the 12 classes of the CamVid dataset:
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
| GFlops            | 75.8180               |
| MParams           | 26.7043               |
| Source framework  | TensorFlow\*          |

## Accuracy

The quality metrics were calculated on the CamVid validation dataset. The `unlabeled` class had been ignored during metrics calculation.

| Metric                    | Value         |
|---------------------------|---------------|
| mIoU                      |        69.54% |

- `IOU=TP/(TP+FN+FP)`, where:
  - `TP` - number of true positive pixels for given class
  - `FN` - number of false negative pixels for given class
  - `FP` - number of false positive pixels for given class

## Input

Image, shape - `1, 3, 720, 960`, format is `B, C, H, W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

Semantic segmentation class prediction map, shape - `1, 720, 960`, output data format is `B, H, W`, where:

- `B` - batch size
- `H` - horizontal coordinate of the input pixel
- `W` - vertical coordinate of the input pixel

Output contains the class prediction result of each pixel.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
