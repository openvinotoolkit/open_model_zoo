# Sphereface

## Use Case and High-Level Description

[Deep face recognition under open-set protocol](https://arxiv.org/pdf/1704.08063.pdf)

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Face recognition |
| GFLOPs            | 3.504         |
| MParams           | 22.671        |
| Source framework  | Caffe\*       |

## Accuracy

## Performance

## Input

### Original model

Image, name - `data`,  shape - `1,3,112,96`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.
Mean values - [127.5,127.5,127.5], scale value - 128

### Converted model

Image, name - `data`,  shape - `1,3,112,96`, format is `B,C,H,W` where:

- `B` - batch size
- `C` - channel
- `H` - height
- `W` - width

Channel order is `BGR`.

## Output

### Original model

Face embedings, name - `fc5`,  shape - `1,512`, output data format  - `B,C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating points values, face embeddings

The net outputs on different images are comparable in cosine distance.

### Converted model

Face embedings, name - `fc5`,  shape - `1,512`, output data format  - `B,C`, where:

- `B` - batch size
- `C` - row-vector of 512 floating points values, face embeddings

The net outputs on different images are comparable in cosine distance.

## Legal Information

[LICENSE](https://raw.githubusercontent.com/wy1iu/sphereface/master/LICENSE)
