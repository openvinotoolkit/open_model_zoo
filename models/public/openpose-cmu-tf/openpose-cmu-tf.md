# openpose-cmu-tf

## Use Case and High-Level Description

This is TensoFlow version of OpenPose algorithm, used for multi-person
human pose estimation. The model is based on VGG pretrained network,
which described in the original [paper](https://arxiv.org/pdf/1812.08008.pdf).
For details see [repository](https://github.com/ildoonet/tf-pose-estimation).
## Example

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| Type              | Human pose estimation |
| GFLOPs            | 319.304               |
| MParams           | 52.311                |
| Source framework  | TensorFlow\*          |

## Accuracy

## Performance

## Input

### Original mode

Image, name - `image`,  shape - `1,368,432,3`, format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - channel

Channel order is `BGR`.

### Converted model

Image, name - `image`,  shape - `1,3,368,432`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Concatenation of keypoint pairwise relations (part affinity fields) and keypoint heatmaps,
name - `Openpose/concat_stage7`, output data format is `B,H,W,C` where:

- `B` - batch size
- `H` - height
- `W` - width
- `C` - heatmap channel

### Converted model

1. Predicted keypoint heatmaps, name - `Mconv7_stage6_L2/Conv2D`, shape - `1,19,46,54`,
   output data format is `B,C,H,W` where:

- `B` - batch size
- `C` - number of predicted body parts
- `H` - height of heatmap
- `W` - width of heatmap

2. Predicted part affinity fields (PAFs) - `Mconv7_stage6_L1/Conv2D`, shape - `1,38,46,54`,
   output data format is `B,C,H,W` where:

- `B` - batch size
- `C` - number of predicted PAFs
- `H` - height of PAF
- `W` - width of PAF

## Legal Information

[https://raw.githubusercontent.com/ildoonet/tf-pose-estimation/master/LICENSE]()