# icnet-camvid-int8-onnx-0001

## Use Case and High-Level Description

This is the ICNet model that is designed to perform semantic segmentation. The model has been pretrained on the CamVid dataset and then quantized to INT8 fixed-point precision using so-called Quantization-aware training approach implemented in PyTorch framework.  Training used median frequency balancing for class weighing and random horizontal flips as data augmentation. For details about the original floating point model, check out the [paper](https://arxiv.org/pdf/1704.08545.pdf).

The model input is a blob that consists of a single image of "1x3x768x768" in BGR order. The pixel values are integers in the [0, 255] range.

The model output for `icnet-camvid-int8-onnx-0001` is the per-pixel probabilities of each input pixel belonging to one of the 12 classes of the CamVid dataset.

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| Type              | Semantic segmentation |
| Source framework  | PyTorch\*             |

## Accuracy

The quality metrics were calculated on the CamVid validation dataset. The 'unlabeled' class had been ignored during metrics calculation.

| Metric                    | Value         |
|---------------------------|---------------|
| mIoU                      |        65.57% |

- `IOU=TP/(TP+FN+FP)`, where:
  - `TP` - number of true positive pixels for given class
  - `FN` - number of false negative pixels for given class
  - `FP` - number of false positive pixels for given class


## Performance

## Input

Image, shape - `1,3,768,768`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`

## Output

Semantic segmentation class probabilities map, shape -`1,12,768,768`, output data format is `B,C,H,W` where:

- `B` - batch size
- `C` - predicted probabilities of input pixel belonging to class `C` in  [0, 1] range
- `H` - horizontal coordinate of the input pixel
- `W` - vertical coordinate of the input pixel

## Legal Information
[*] Other names and brands may be claimed as the property of others.
