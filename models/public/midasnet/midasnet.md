# midasnet

## Use Case and High-Level Description

MidasNet is a model for monocular depth estimation trained by mixing several datasets;
as described in the following paper:
"Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer"
<https://arxiv.org/pdf/1907.01341>

The model input is a blob that consists of a single image of "1x3x384x384" in RGB order.

The model output is an inverse depth map that is defined up to an unknown scale factor.

## Example

See [here](https://github.com/intel-isl/MiDaS)

## Legal Information

[LICENSE](https://drive.google.com/open?id=1p_7P7VKSpD1xM8Ex6p0epZ4TdYFPYjss)
