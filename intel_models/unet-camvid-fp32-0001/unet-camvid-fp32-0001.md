# unet-camvid-fp32-0001

## Use Case and High-Level Description

This is the UNet model that is designed to perform semantic segmentation. The model has been trained on the CamVid dataset from scratch using PyTorch framework. Training used median frequency balancing for class weighing. For details about the original floating point model, check out the [paper](https://arxiv.org/pdf/1505.04597.pdf).

The model input is a blob that consists of a single image of "1x3x368x480" in RGB order. The pixel values are floating-point values that were obtained by taking each channel value as an integer in the [0, 255] range, dividing it by 255, and the normalizing the results given a mean of [0.39068785, 0.40521392, 0.41434407] and standard deviation of [0.29652068, 0.30514979, 0.30080369] (RGB order).

The model output for `unet-camvid-fp32-0001` is the per-pixel probabilities of each input pixel belonging to one of the 12 classes of the CamVid dataset.

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| Type              | Semantic segmentation |
| Source framework  | PyTorch               |

## Accuracy

The quality metrics calculated on CamVid validation dataset is 71.95% mIoU.

| Metric                    | Value         |
|---------------------------|---------------|
| mIoU                      |        71.95% |

## Performance

## Input

Image, shape - `1,3,368,480`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`

## Output

Semantic segmentation class probabilities map, shape -`1,12,368,480`, output data format is `B,C,H,W` where:

- `B` - batch size
- `C` - predicted probabilities of input pixel belonging to class `C` in  [0, 1] range
- `H` - horizontal coordinate of the input pixel
- `W` - vertical coordinate of the input pixel
