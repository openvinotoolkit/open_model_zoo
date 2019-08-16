# face-recognition-resnet100-arcface

## Use Case and High-Level Description

[Deep face recognition net with ResNet100 backbone and Arcface loss](https://arxiv.org/abs/1801.07698)

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Face recognition |
| GFLOPs            | 24.209        |
| MParams           | 65.131        |
| Source framework  | mxnet\*       |

## Accuracy

## Performance

## Input

### Original model

Image, name - `data`,  shape - `1,3,112,112`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

### Converted model

Image, name - `data`,  shape - `1,3,112,112`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Face embedings, name - `pre_fc1`,  shape - `1,512`, output data format  - `B,C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating points values, face embeddings

The net outputs on different images are comparable in cosine distance.

### Converted model

Face embedings, name - `pre_fc1`,  shape - `1,512`, output data format  - `B,C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating points values, face embeddings

The net outputs on different images are comparable in cosine distance.

## Legal Information

[LICENSE](https://raw.githubusercontent.com/deepinsight/insightface/master/LICENSE)
