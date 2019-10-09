# face-recognition-mobilefacenet-arcface

The original name of the model is [MobileFaceNet,ArcFace@ms1m-refine-v1](https://github.com/deepinsight/insightface/wiki/Model-Zoo).

## Use Case and High-Level Description

[Deep face recognition net with MobileFaceNet backbone and Arcface loss](https://arxiv.org/abs/1801.07698)

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Face recognition |
| GFLOPs            | 0.449         |
| MParams           | 0.993         |
| Source framework  | MXNet\*       |

## Accuracy

## Performance

## Input

### Original Model

Image, name: `data`,  shape: `1,3,112,112`, format is `B,C,H,W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `RGB`.

### Converted Model

Image, name: `data`,  shape: `1,3,112,112`, format: `B,C,H,W`, where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original Model

Face embedings, name: `pre_fc1`,  shape: `1,512`, output data format: `B,C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating points values, face embeddings

The net outputs on different images are comparable in cosine distance.

### Converted Model

Face embeddings, name: `pre_fc1`,  shape: `1,512`, output data format: `B,C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating points values, face embeddings

The net outputs on different images are comparable in cosine distance.

## Legal Information

[LICENSE](https://raw.githubusercontent.com/deepinsight/insightface/master/LICENSE)