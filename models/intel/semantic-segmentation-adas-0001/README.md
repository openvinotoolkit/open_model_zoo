# semantic-segmentation-adas-0001

## Use Case and High-Level Description

This is a segmentation network to classify each pixel into 20 classes:
- road
- sidewalk
- building
- wall
- fence
- pole
- traffic light
- traffic sign
- vegetation
- terrain
- sky
- person
- rider
- car
- truck
- bus
- train
- motorcycle
- bicycle
- ego-vehicle

## Example

![](./assets/semantic-segmentation-adas-0001.png)

## Specification

| Metric          | Value     |
|---------------- |---------- |
| Image size      | 2048x1024 |
| GFlops          | 58.572    |
| MParams         | 6.686     |
| Source framework| Caffe\*   |

## Accuracy

The quality metrics calculated on 2000 images:

| Label         | IOU         |
|---------------|-------------|
| **mean**      |  **0.6907** |
| Road          |    0.910379 |
| Sidewalk      |    0.630676 |
| Building      |    0.860139 |
| Wall          |    0.424166 |
| Fence         |    0.592632 |
| Pole          |    0.559078 |
| Traffic Light |    0.654779 |
| Traffic Sign  |    0.648217 |
| Vegetation    |    0.882593 |
| Terrain       |    0.620521 |
| Sky           |    0.976889 |
| Person        |    0.711653 |
| Rider         |    0.612787 |
| Car           |    0.877892 |
| Truck         |    0.674829 |
| Bus           |    0.743752 |
| Train         |    0.358641 |
| Motorcycle    |    0.600701 |
| Bicycle       |    0.622246 |
| Ego-Vehicle   |    0.852932 |

- `IOU=TP/(TP+FN+FP)`, where:
  - `TP` - number of true positive pixels for given class
  - `FN` - number of false negative pixels for given class
  - `FP` - number of false positive pixels for given class

## Inputs

The blob with `BGR` image and the shape `1, 3, 1024, 2048` in the format `B, C, H, W`, where:

- `B` – batch size
- `C` – number of channels
- `H` – image height
- `W` – image width

## Outputs

The net output is a blob with the shape `1, 1, 1024, 2048` in the format `B, C, H, W`. It can be treated as a
one-channel feature map, where each pixel is a label of one of the classes.

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Image Segmentation C++ Demo](../../../demos/segmentation_demo/cpp/README.md)
* [Image Segmentation Python\* Demo](../../../demos/segmentation_demo/python/README.md)
* [Whiteboard Inpainting Python\* Demo](../../../demos/whiteboard_inpainting_demo/python/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
