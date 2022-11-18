# image-retrieval-0001

## Use Case and High-Level Description

Image retrieval model based on [MobileNetV2](https://arxiv.org/abs/1801.04381) architecture as a backbone.

## Example

![](./assets/image-retrieval-0001.jpg)

## Specification

| Metric                                                        | Value                   |
|---------------------------------------------------------------|-------------------------|
| Top1 accuracy                                                 | 0.834                   |
| GFlops                                                        | 0.613                   |
| MParams                                                       | 2.535                   |
| Source framework                                              | TensorFlow\*            |

## Inputs

Image, name: `Placeholder`, shape: `1, 224, 224, 3` in the format `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `BGR`.

## Outputs

Tensor with name `model/tf_op_layer_l2_normalize/l2_normalize` and the shape `1, 256` — image embedding vector.

## Demo usage

The model can be used in the following demos provided by the Open Model Zoo to show its capabilities:

* [Image Retrieval Python\* Demo](../../../demos/image_retrieval_demo/python/README.md)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
