# horizontal-text-detection-0001

## Use Case and High-Level Description

Text detector based on [FCOS](https://arxiv.org/abs/1904.01355) architecture with [MobileNetV2-like](https://arxiv.org/abs/1801.04381) as a backbone for indoor/outdoor scenes with more or less horizontal text.

The key benefit of this model compared to the [base model](../text-detection-0003/README.md) is its smaller size and faster performance.

## Example

![](./assets/horizontal-text-detection-0001.png)

## Specification

| Metric                                                        | Value     |
|---------------------------------------------------------------|-----------|
| F-measure (harmonic mean of precision and recall on ICDAR2013)| 88.45%    |
| GFlops                                                        | 7.78      |
| MParams                                                       | 2.26      |
| Source framework                                              | PyTorch\* |

## Inputs

Image, name: `image`, shape: `1, 3, 704, 704` in the format `1, C, H, W`, where:

- `C` - number of channels
- `H` - image height
- `W` - image width

Expected color order - `BGR`.

## Outputs

1. The `boxes` is a blob with the shape `100, 5` in the format `N, 5`, where `N` is the number of detected
   bounding boxes. For each detection, the description has the format:
   [`x_min`, `y_min`, `x_max`, `y_max`, `conf`], where:

    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner
    - `conf` - confidence for the predicted class

2. The `labels` is a blob with the shape `100` in the format `N`, where `N` is the number of detected
   bounding boxes. In case of text detection, it is equal to `0` for each detected box.

## Training Pipeline

The OpenVINO [Training Extensions](https://github.com/openvinotoolkit/training_extensions/blob/misc/README.md) provide a [training pipeline](https://github.com/openvinotoolkit/training_extensions/blob/misc/models/object_detection/model_templates/horizontal-text-detection/readme.md), allowing to fine-tune the model on custom dataset.

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Text Detection C++ Demo](../../../demos/text_detection_demo/cpp/README.md)

## Legal Information

[*] Other names and brands may be claimed as the property of others.
