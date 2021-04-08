# faster-rcnn-resnet101-coco-sparse-60-0001

## Use Case and High-Level Description

This is a retrained version of the [Faster R-CNN](https://arxiv.org/abs/1506.01497) object detection network trained with the [Common Objects in Context (COCO)](https://cocodataset.org/#home) training dataset.
The actual implementation is based on [Detectron](https://github.com/facebookresearch/detectron2),
with additional [network weight pruning](https://arxiv.org/abs/1710.01878) applied to sparsify convolution layers (60% of network parameters are set to zeros).

The model input is a blob that consists of a single image of `1, 3, 800, 1280` in the `BGR` order. The pixel values are integers in the [0, 255] range.

## Specification

| Metric                       | Value        |
|------------------------------|--------------|
| Mean Average Precision (mAP) | 38.74%\**    |
| GFlops                       | 849.9109     |
| MParams                      | 52.79        |
| Source framework             | TensorFlow\* |

See Average Precision metric description at [COCO: Common Objects in Context](https://cocodataset.org/#detection-eval). The primary challenge metric is used. Tested on the COCO validation dataset.

## Inputs

Image, name: `input`, shape: `1, 3, 800, 1280` in the format `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order is `BGR`.

## Outputs

The net outputs a blob with the shape `300, 7`, where each row consists of [`image_id`, `class_id`, `confidence`, `x0`, `y0`, `x1`, `y1`] respectively:

- `image_id` - image ID in the batch
- `class_id` - predicted class ID in range [1, 80], mapping to class names provided in `<omz_dir>/data/dataset_classes/coco_80cl_bkgr.txt` file
- `confidence` - [0, 1] detection score; the higher the value, the more confident the detection is
- (`x0`, `y0`) - normalized coordinates of the top left bounding box corner, in the [0, 1] range
- (`x1`, `y1`) - normalized coordinates of the bottom right bounding box corner, in the [0, 1] range

## Legal Information
[\*] Other names and brands may be claimed as the property of others.

[\**] May be different from the original implementation due to different input configurations.
