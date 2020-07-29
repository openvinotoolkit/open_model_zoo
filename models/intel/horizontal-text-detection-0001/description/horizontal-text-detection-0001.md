# text-detection-0003

## Use Case and High-Level Description

Text detector based on [FCOS](https://arxiv.org/abs/1904.01355) architecture with [MobileNetV2-like](https://arxiv.org/abs/1801.04381) as a backbone for indoor/outdoor scenes with more or less horizontal text.

The key benefit of this model compared to the [base model](../../text-detection-0003/description/text-detection-0003.md) is its smaller size and faster performance.
## Example

![](./horizontal-text-detection-0001.png)

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| F-measure (harmonic mean of precision and recall on ICDAR2013)| 88.45%                 |
| GFlops                                                        | 7.78                  |
| MParams                                                       | 2.26                   |
| Source framework                                              | PyTorch\*              |


## Performance

## Inputs

1. Name: `input`, shape: [1x3x704x704] - An input image in the format [1xCxHxW],
   where:

    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

## Outputs

1. The `boxes` is a blob with shape: [N, 5], where N is the number of detected
   bounding boxes. For each detection, the description has the format:
   [`x_min`, `y_min`, `x_max`, `y_max`, `conf`],
   where:
    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner.
    - `conf` - confidence for the predicted class
2. The `labels` is a blob with shape: [N], where N is the number of detected
   bounding boxes. In case of text detection, it is equal to `0` for each detected box.


## Legal Information
[*] Other names and brands may be claimed as the property of others.
