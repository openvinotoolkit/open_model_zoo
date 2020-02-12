# midasnet

## Use Case and High-Level Description

MidasNet is a model for monocular depth estimation trained by mixing several datasets;
as described in the following paper:
"Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer"
<https://arxiv.org/pdf/1907.01341>

The model input is a blob that consists of a single image of "1x3x384x384" in `BGR` order.

The model output is an inverse depth map that is defined up to an unknown scale factor.

## Example

See [here](https://github.com/intel-isl/MiDaS)

## Input

Image, name - `data`, shape - `1,3,384,384`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

Inverse depth map, name - `data`, shape - `1,384,384`, format is `B,H,W` where:

- `B` - batch size
- `H` - height
- `W` - width

Inverse depth map is defined up to an unknown scale factor.

## Legal Information

[LICENSE](https://drive.google.com/open?id=1p_7P7VKSpD1xM8Ex6p0epZ4TdYFPYjss)
