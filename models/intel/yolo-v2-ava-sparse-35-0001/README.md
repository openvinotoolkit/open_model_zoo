# yolo-v2-ava-sparse-35-0001

## Use Case and High-Level Description

This is a reimplemented and retrained version of the [YOLO v2](https://arxiv.org/abs/1612.08242) object detection network trained with the VOC2012 training dataset.
[Network weight pruning](https://arxiv.org/abs/1710.01878) is applied to sparsify convolution layers (35% of network parameters are set to zeros).

## Specification

| Metric                       | Value        |
|------------------------------|--------------|
| Mean Average Precision (mAP) | 63.71%       |
| GFlops                       | 29.4205      |
| MParams                      | 50.6451      |
| Source framework             | TensorFlow\* |

For Average Precision metric description, see [The PASCAL Visual Object Classes (VOC) Challenge](https://link.springer.com/article/10.1007/s11263-009-0275-4).
Tested on the VOC 2012 validation dataset.

## Inputs

Image, name: `data`, shape: `1, 416, 416, 3` in the format `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order is `BGR`.

## Outputs

The net outputs a blob with the shape `1, 21125` which can be reshaped to `5, 25, 13, 13`,
where each number corresponds to [`num_anchors`, `cls_reg_obj_params`, `y_loc`, `x_loc`] respectively:

- `num_anchors`: number of anchor boxes, each spatial location specified by `y_loc` and `x_loc` has five anchors
- `cls_reg_obj_params`: parameters for classification and regression. The values are made up of the following:
  * Regression parameters (4)
  * Objectness score (1)
  * Class score (20), mapping to class names provided by `<omz_dir>/data/dataset_classes/voc_20cl.txt` file.
- `y_loc` and `x_loc`: spatial location of each grid

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Object Detection C++ Demo](../../../demos/object_detection_demo/cpp/README.md)
* [Object Detection Python\* Demo](../../../demos/object_detection_demo/python/README.md)
* [Pedestrian Tracker C++ Demo](../../../demos/pedestrian_tracker_demo/cpp/README.md)

## Legal Information
[*] Same as the original implementation.

[**] Other names and brands may be claimed as the property of others.
