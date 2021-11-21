# ocrnet-hrnet-w48-paddle

## Use Case and High-Level Description

`ocrnet-hrnet-w48-paddle` is a semantic segmentation model, pre-trained on on [Cityscapes](https://www.cityscapes-dataset.com) dataset for 19 object classes, listed in `<omz_dir>/data/dataset_classes/cityscapes_19cl_bkgr.txt` file. See Cityscapes classes [definition](https://www.cityscapes-dataset.com/dataset-overview) for more details. The model was built on [HRNet](https://arxiv.org/abs/1904.04514) backbone and address the semantic segmentation problem characterizing a pixel by exploiting the representation of the corresponding object class using Object-Contextual Representations. This model is used for pixel-level prediction tasks. For details see [repository](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.3/configs/ocrnet), [paper](https://arxiv.org/abs/1909.11065).

## Specification

| Metric            | Value                |
|-------------------|----------------------|
| Type              | Semantic segmentation|
| GFlops            | TBD                  |
| MParams           | TBD                  |
| Source framework  | Paddle\*             |

## Accuracy

| Metric    | Value |
| --------- | ----- |
| mean_iou  | TBD   |

Accuracy metrics were obtained with fixed input resolution 2048x1024.

## Input

### Original model

Image, name: `x`, shape: `1, 3, 1024, 2048`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `RGB`.
Mean values: [127.5, 127.5, 127.5], scale values: [127.5, 127.5, 127.5]

### Converted Model

Image, name: `x`, shape: `1, 3, 1024, 2048`, format: `B, C, H, W`, where:

- `B` - batch size
- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order: `BGR`.

## Output

### Original Model

Integer values in a range [0, 18], which represent an index of a predicted class for each image pixel. Name: `argmax_0.tmp_0`, shape: `1, 1024, 2048` in `B, H, W` format, where:

- `B` - batch size
- `H` - image height
- `W` - image width

### Converted Model

Integer values in a range [0, 18], which represent an index of a predicted class for each image pixel. Name: `argmax_0.tmp_0`, shape: `1, 1024, 2048` in `B, H, W` format, where:

- `B` - batch size
- `H` - image height
- `W` - image width

## Download a Model and Convert it into Inference Engine Format

You can download models and if necessary convert them into Inference Engine format using the [Model Downloader and other automation tools](../../../tools/model_tools/README.md) as shown in the examples below.

An example of using the Model Downloader:
```
omz_downloader --name <model_name>
```

An example of using the Model Converter:
```
omz_converter --name <model_name>
```

## Legal Information

The original model is distributed under the
[Apache License, Version 2.0](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.3/LICENSE).
