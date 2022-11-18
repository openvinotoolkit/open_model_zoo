# icnet-camvid-ava-sparse-30-0001

## Use Case and High-Level Description

A trained model of ICNet for fast semantic segmentation, trained on the CamVid dataset from scratch using the TensorFlow\* framework. The trained model has 30% sparsity (ratio of zeros within all the convolution kernel weights). For details about the original floating-point model, check out the [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545).

The model input is a blob that consists of a single image of `1, 720, 960, 3` in the `BGR` order. The pixel values are integers in the [0, 255] range.

The model output for `icnet-camvid-ava-sparse-30-0001` is the predicted class index of each input pixel belonging to one of the 12 classes of the CamVid dataset:
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
| mIoU                      |        75.87% |

- `IOU=TP/(TP+FN+FP)`, where:
  - `TP` - number of true positive pixels for given class
  - `FN` - number of false negative pixels for given class
  - `FP` - number of false positive pixels for given class

## Input

Image, name: `data`, shape - `1, 720, 960, 3`, format is `B, H, W, C`, where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

## Output

Semantic segmentation class prediction map, shape - `1, 720, 960`, output data format is `B, H, W`, where:

- `B` - batch size
- `H` - horizontal coordinate of the input pixel
- `W` - vertical coordinate of the input pixel

Output contains the class prediction result of each pixel.

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Image Segmentation C++ Demo](../../../demos/segmentation_demo/cpp/README.md)
* [Image Segmentation Python\* Demo](../../../demos/segmentation_demo/python/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
