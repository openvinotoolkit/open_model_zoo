# product-detection-0001

## Use Case and High-Level Description

Product detector based on SSD-lite architecture with [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf) as a backbone for Self-checkout Point of Sale related scenes.
The detector is able to detect 12 classes of objects (`sprite`, `kool-aid`, `extra`, `ocelo`, `finish`, `mtn_dew`, `best_foods`, `gatorade`, `heinz`, `ruffles`, `pringles`, `del_monte`) and first labels 0 and 1 are related to `background_label` and `undefined` correspondingly.

## Example

`TBD`

## Specification

| Metric                                                            | Value    |
|-------------------------------------------------------------------|----------|
| Average Precision (AP) @[ IoU=0.50:0.95,  area=all, maxDets=100 ] | 0.715    |
| GFlops                                                            | 3.598    |
| MParams                                                           | 3.212    |
| Source framework                                                  | Pytorch* |

## Performance

## Inputs

1. name: "input" , shape: [1x3x512x512] - An input image in the format [BxCxHxW],
   where:

    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   Expected color order - BGR.

## Outputs

1. The net outputs a blob with shape: [1, 1, N, 7], where N is the number of detected
   bounding boxes. For each detection, the description has the format:
   [`image_id`, `label`, `conf`, `x_min`, `y_min`, `x_max`, `y_max`],
   where:

    - `image_id` - ID of the image in the batch
    - `label` - predicted class ID
    - `conf` - confidence for the predicted class
    - (`x_min`, `y_min`) - coordinates of the top left bounding box corner
    - (`x_max`, `y_max`) - coordinates of the bottom right bounding box corner.

## Legal Information
[*] Other names and brands may be claimed as the property of others.
